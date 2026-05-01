# Round 2 -- Remaining LOW-Severity Bugs

Checked: 2026-03-17

## Bug 1: Debye length uses `potential_scale` instead of `thermal_voltage`
- **Status:** STILL PRESENT
- **File:** `/Nondim/transform.py`, line 399-402
- **Detail:** The Debye length formula `sqrt(eps * potential_scale / (F * c_ref))` uses the user-configurable `potential_scale` instead of the physical `thermal_voltage_v`. The Debye length is a physical constant that should always use V_T = RT/F. When `potential_scale_v` equals V_T (the default), this is harmless. If a user sets a different `potential_scale_v`, the reported Debye length (and `debye_to_length_ratio`) will be wrong. The Poisson coefficient on line 395-397 correctly uses `potential_scale` (that is the nondim coefficient, not a physical constant), so only the Debye length diagnostic is affected.
- **Severity:** LOW (diagnostic only, does not affect PDE assembly)

## Bug 2: Default stoichiometry `[-1] * n_species`
- **Status:** STILL PRESENT
- **File:** `/Forward/bv_solver/config.py`, line 24
- **Detail:** `stoichiometry` defaults to `[-1] * n_species`, meaning every species is consumed at the electrode. For multi-species problems (e.g., ORR with O2 and H2O2), some species are products (stoichiometry +1). Defaulting all to -1 is chemically incorrect for any multi-species reaction where at least one species is produced. Users must always override this, but the silent default can mask configuration errors.
- **Severity:** LOW (users of BV solver must configure stoichiometry anyway)

## Bug 3: Robin solver zero time steps
- **Status:** FIXED
- **File:** `/Forward/robin_solver.py`, line 279
- **Detail:** Now uses `max(1, int(round(t_end_model / dt_model)))`, preventing zero-step runs. Confirmed fixed.

## Bug 4: Full options dict passed as `solver_parameters`
- **Status:** STILL PRESENT
- **File:** `/Forward/bv_solver/solvers.py`, lines 58, 124, 308, 463
- **Detail:** `params` is `solver_params[10]`, which is a dict containing PETSc keys (`snes_type`, `ksp_type`, etc.) mixed with application keys (`robin_bc`, `nondim`, `bv_bc`). This entire dict is passed to `fd.NonlinearVariationalSolver(problem, solver_parameters=params)`. FireDrake expects only PETSc-recognized keys. The extra keys (`robin_bc`, `nondim`, `bv_bc`) are silently ignored by PETSc but this is poor practice -- it relies on PETSc not erroring on unknown keys, and future PETSc versions could start rejecting them. The application-level keys should be stripped before passing to the solver.
- **Severity:** LOW (works today because PETSc ignores unknown keys)

## Bug 5: `build_solver_options` hardcodes 2-species
- **Status:** STILL PRESENT
- **File:** `/Nondim/compat.py`, lines 53-55
- **Detail:** `robin_bc.kappa` is hardcoded as `[0.8, 0.8]` and `c_inf` as a 2-element list. This function only works for 2-species problems. No `n_species` parameter is accepted. Any caller with a different number of species must build their own options dict.
- **Severity:** LOW (convenience function, callers can bypass)

## Bug 6: `write_phi_applied_flux_csv` crashes on bare filename
- **Status:** STILL PRESENT
- **File:** `/Forward/steady_state/common.py`, line 222
- **Detail:** `os.makedirs(os.path.dirname(csv_path), exist_ok=True)` crashes with `FileNotFoundError` when `csv_path` is a bare filename like `"results.csv"` because `os.path.dirname("results.csv")` returns `""`, and `os.makedirs("")` raises an error. Fix: guard with `dirname = os.path.dirname(csv_path); if dirname: os.makedirs(dirname, exist_ok=True)`.
- **Severity:** LOW (only triggers if caller passes a bare filename without directory)

## Bug 7: Dirichlet phi0 bounds prevent negative potentials
- **Status:** STILL PRESENT
- **File:** `/Inverse/parameter_targets.py`, line 153-154
- **Detail:** `default_bounds_factory` returns `(1e-8, None)` for the `phi0` parameter target. The lower bound of `1e-8` prevents phi0 from being zero or negative. For electrochemical systems where the bulk/reference potential can be negative (e.g., vs. different reference electrodes), this bound is physically wrong. Should be `(None, None)` or at minimum allow negative values.
- **Severity:** LOW (only affects users relying on default bounds for phi0 inference)

## Bug 8: `EnsembleMeanWrapper` ddof=1 vs `predict_ensemble` ddof=0
- **Status:** STILL PRESENT
- **File:** `/Surrogate/ensemble.py`, lines 100-102 vs `/Surrogate/nn_training.py`, lines 681-683
- **Detail:** Two ensemble prediction paths compute standard deviation with different conventions:
  - `EnsembleMeanWrapper._predict_ensemble_raw()` uses `ddof=1` (sample std, Bessel correction)
  - `predict_ensemble()` in `nn_training.py` uses `ddof=0` (population std, default numpy)
  Both are used for uncertainty estimation. With typical ensemble sizes of 5 members, the difference is ~12% (sqrt(5/4) = 1.118). This inconsistency means uncertainty estimates differ depending on which code path is used.
- **Severity:** LOW (12% difference in uncertainty band width for 5 members)

## Bug 9: Stale v11 backup files in `data/surrogate_models/`
- **Status:** STILL PRESENT
- **Files:**
  - `data/surrogate_models/split_indices_v11_backup.npz`
  - `data/surrogate_models/training_data_merged_v11_backup.npz`
- **Detail:** These backup files from an earlier version are still present. They increase repo size and could cause confusion about which data is canonical. They appear in git status as untracked files.
- **Severity:** LOW (clutter only, no functional impact)

## Bug 10: K-fold CV no shuffle
- **Status:** STILL PRESENT
- **File:** `/Surrogate/pod_rbf_model.py`
- **Detail:** No `shuffle` parameter found in `KFold` usage (in fact, no `KFold` import found at all in this file currently). The file has no shuffle-related code. If cross-validation is performed elsewhere or was removed, the original concern about ordered data biasing folds may no longer apply. However, inspecting the file shows no CV code at all now, meaning this functionality may have been removed or relocated.
- **Severity:** LOW (may be moot if CV was removed from this file)

## Bug 11: Dead `_make_sp_mms` function
- **Status:** STILL PRESENT
- **File:** `/scripts/verification/mms_bv_4species.py`, line 128
- **Detail:** `_make_sp_mms()` is defined but never called. The script uses `_make_sp_mms_fixed()` (line 400) instead. The original function is dead code that should be removed.
- **Severity:** LOW (dead code, no functional impact)

## Bug 12: Hardcoded `N_WORKERS=8`
- **Status:** STILL PRESENT
- **File:** `/scripts/surrogate/overnight_train_v11.py`, line 57
- **Detail:** `N_WORKERS = 8` is hardcoded. On machines with fewer cores, this wastes resources or causes contention. On machines with more cores, it underutilizes capacity. Should use `os.cpu_count()` or accept a CLI argument.
- **Severity:** LOW (training script, not production code)

---

## Summary

| # | Bug | Status |
|---|-----|--------|
| 1 | Debye length uses potential_scale | STILL PRESENT |
| 2 | Default stoichiometry [-1]*n | STILL PRESENT |
| 3 | Robin solver zero time steps | FIXED |
| 4 | Full options dict as solver_parameters | STILL PRESENT |
| 5 | build_solver_options hardcodes 2-species | STILL PRESENT |
| 6 | write_phi_applied_flux_csv empty dirname | STILL PRESENT |
| 7 | Dirichlet phi0 bounds prevent negative | STILL PRESENT |
| 8 | EnsembleMeanWrapper ddof=1 vs ddof=0 | STILL PRESENT |
| 9 | Stale v11 backup files | STILL PRESENT |
| 10 | K-fold CV no shuffle | STILL PRESENT (possibly moot) |
| 11 | Dead _make_sp_mms function | STILL PRESENT |
| 12 | Hardcoded N_WORKERS=8 | STILL PRESENT |

**Total: 1 FIXED, 11 STILL PRESENT, 0 WORSENED**

No new LOW bugs discovered during this review.
