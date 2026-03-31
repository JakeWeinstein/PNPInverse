# Round 4 Comprehensive Bug Audit — 2026-03-31

**Method:** 6 parallel agents, each auditing a distinct subsystem of the v13 master pipeline.
**Scope:** All dependencies of `Infer_BVMaster_charged_v13_ultimate.py` — ~80 Python files across Forward/, FluxCurve/, Surrogate/, Inverse/, Nondim/, scripts/, and tests/.

---

## Executive Summary

| Severity | Count | Breakdown |
|----------|-------|-----------|
| **CRITICAL** | **0** | All prior CRITICALs confirmed fixed or downgraded |
| **HIGH** | **5** | 2 test infrastructure, 1 diagnostic, 1 data leakage, 1 numerical |
| **MEDIUM** | **14** | Across all subsystems |
| **LOW** | **19** | Maintenance, performance, documentation |
| **Total** | **38** | |

**Key positive finding:** The core mathematical foundations are sound — nondimensionalization, Butler-Volmer kinetics, adjoint gradients, surrogate normalization, and the inference algorithm itself are all correct. Most prior CRITICAL/HIGH bugs have been properly fixed.

---

## HIGH Severity Bugs (Fix Before Next Publication)

### H1. Broken import paths in `test_pipeline_reproducibility.py`
- **Agent:** 6 (Pipeline Integration)
- **File:** `tests/test_pipeline_reproducibility.py:257,384`
- **Error:** Script moved from `scripts/surrogate/` to `scripts/Inference/` but test still references old path. The entire PIP-01 regression test suite is non-functional.
- **Fix:** Update import path to `scripts.Inference.Infer_BVMaster_charged_v13_ultimate`.

### H2. Wrong voltage grid in `test_pipeline_reproducibility.py`
- **Agent:** 6 (Pipeline Integration)
- **File:** `tests/test_pipeline_reproducibility.py:283`
- **Error:** Test concatenates only `eta_symmetric + eta_shallow` (16 points) but pipeline uses all three grids including `eta_cathodic` (22 points). Stored baselines are invalid.
- **Fix:** Add `eta_cathodic` to the test's `all_eta` computation.

### H3. `last_reason` uninitialized in BV point-solve attempt loop
- **Agent:** 2 (FluxCurve)
- **File:** `FluxCurve/bv_point_solve/__init__.py:779`
- **Error:** `'last_reason' in dir()` is fragile; `last_reason` not initialized before the attempt loop. Failed points always report generic message instead of actual failure reason.
- **Fix:** Add `last_reason = "all attempts failed"` before the loop.

### H4. ISMO `_retrain_surrogate` data leakage pathway
- **Agent:** 5 (ISMO/Data)
- **File:** `Surrogate/ismo.py:970-983`
- **Error:** Creates its own 80/20 train/val split from full dataset without respecting canonical `split_indices.npz` test indices. Test points can leak into training. Only affects `run_ismo_live.py` path (the `run_ismo.py` path correctly delegates to `ismo_retrain.retrain_nn_ensemble`).
- **Fix:** Pass through canonical `train_idx`/`test_idx` and only split within `train_idx`.

### H5. Dirichlet solver extreme condition number (~1e14)
- **Agent:** 4 (PDE Solvers)
- **File:** `Forward/dirichlet_solver.py:163-165`
- **Error:** Poisson equation assembled with raw SI permittivity (~7e-10 F/m) against Faraday constant (~96485 C/mol). Condition number ratio ~1e14 makes solver numerically intractable for charged species without pre-nondimensionalization. The Robin and BV solvers handle this correctly via `build_model_scaling`.
- **Fix:** Document that inputs must be pre-scaled, or add nondimensionalization.

---

## MEDIUM Severity Bugs

### M1. R-space target volume integral (latent, masked by unit mesh)
- **Agent:** 2 (FluxCurve)
- **File:** `FluxCurve/bv_point_solve/__init__.py:659-665`
- **Error:** BV sequential path uses `fd.assemble(target_ctrl * dx)` = `target * Volume(mesh)`. Currently masked because mesh volume = 1.0. Other paths correctly use `fd.Constant(float(target_flux))`.
- **Fix:** Replace with `fd.Constant(float(target_i))`.

### M2. Fail-penalty gradient can destabilize optimizer
- **Agent:** 2 (FluxCurve)
- **File:** `FluxCurve/bv_point_solve/__init__.py:758-769`
- **Error:** Failed points add ~`1e7 * sign(ctrl)` to gradient (from `fail_penalty=1e9`). When many points fail, gradient is dominated by penalty, causing overshooting.
- **Fix:** Decouple fail-gradient magnitude from `fail_penalty`.

### M3. Multi-pH condition mutates nested reaction config dicts
- **Agent:** 2 (FluxCurve)
- **File:** `FluxCurve/bv_curve_eval.py:468-473`
- **Error:** In-place mutation of `cathodic_conc_factors` dicts inside deep-copied request. Fragile if deepcopy is incomplete.

### M4. PCE `predict_batch` wrong shape for single-sample input
- **Agent:** 3 (Surrogate)
- **File:** `Surrogate/pce_model.py:499-504`
- **Error:** Returns `(n_eta,)` instead of `(1, n_eta)` for single-row input. Breaks downstream batch-size inference.
- **Fix:** Add `cd = np.atleast_2d(cd)` before return.

### M5. NN model mutation in `predict_torch` not exception-safe
- **Agent:** 3 (Surrogate)
- **File:** `Surrogate/nn_model.py:581-588`
- **Error:** Calls `self._model.double()` then `.float()`. If exception occurs between them, model permanently stays in float64.
- **Fix:** Use tensor-level casting instead of mutating model state.

### M6. POD-RBF k-fold CV does not shuffle data
- **Agent:** 3 (Surrogate)
- **File:** `Surrogate/pod_rbf_model.py:262-308`
- **Error:** Consecutive fold splits can be biased if training data has systematic ordering.
- **Fix:** Add random permutation before splitting.

### M7. `retrain_surrogate_full` PCE dispatch unreachable when GPyTorch installed
- **Agent:** 3 (Surrogate)
- **File:** `Surrogate/ismo_retrain.py:1207-1234`
- **Error:** PCE isinstance check is nested inside GP try/except. If GPyTorch is installed, PCE retraining crashes with TypeError.
- **Fix:** Check PCE type independently of GP import success.

### M8. Inconsistent NRMSE normalization between `ismo_pde_eval` and `ismo_convergence`
- **Agent:** 5 (ISMO/Data)
- **File:** `Surrogate/ismo_pde_eval.py:354-389` vs `Surrogate/ismo_convergence.py:367-372`
- **Error:** Same metric name, different normalization (per-sample ptp vs single-curve ptp). Convergence thresholds have different meanings depending on code path.

### M9. `make_standard_pde_bundle` uses global ptp for reference ranges
- **Agent:** 5 (ISMO/Data)
- **File:** `Surrogate/ismo_pde_eval.py:180-181`
- **Error:** `np.ptp(data["current_density"])` flattens across all samples. The 1% NRMSE floor is set very high, potentially masking errors on flat-response samples.

### M10. `k0_2_sensitivity_weight` is a no-op in acquisition
- **Agent:** 5 (ISMO/Data)
- **File:** `Surrogate/acquisition.py`
- **Error:** Creates uniform `np.ones(n_candidates)` multiplier. Users see a tunable parameter that does nothing.

### M11. `run_multi_seed_v13.py` stale-CSV raises RuntimeError (crashes loop)
- **Agent:** 6 (Pipeline Integration)
- **File:** `scripts/studies/run_multi_seed_v13.py:211-216`
- **Error:** Should return `None` on failure, not crash the entire multi-seed assessment.
- **Fix:** Replace `raise RuntimeError(...)` with log + `return None`.

### M12. PDE phases regenerate targets independently from surrogate targets
- **Agent:** 6 (Pipeline Integration)
- **File:** `scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py:1004-1045`
- **Error:** P1/P2 regenerate targets via different solver path (no warm-start) than surrogate targets. Can produce numerically different values at overlapping voltage points.

### M13. PDE phase bounds inconsistent with surrogate bounds
- **Agent:** 6 (Pipeline Integration)
- **File:** `scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py:1027-1028`
- **Error:** PDE uses `k0_lower=1e-8, k0_upper=100.0` while surrogate training bounds are `[1.26e-5, 0.126]` — 3 orders of magnitude wider on both sides.

### M14. Legacy `configure_bv_solver_params` loses k0/alpha mutations
- **Agent:** 4 (PDE Solvers)
- **File:** `Forward/steady_state/bv.py:108,123`
- **Error:** `p.get("bv_bc", {})` returns new dict when key absent; mutations lost.
- **Fix:** Change to `p.setdefault("bv_bc", {})`.

---

## LOW Severity Bugs (19 total)

| ID | Agent | File | Description |
|----|-------|------|-------------|
| L1 | 1 | `_bv_common.py:80` | Gas constant truncated (3e-7 relative error vs Nondim/constants.py) |
| L2 | 1 | `_bv_common.py:99` | Comment says "pH 4 -> 10^-1 M" should be "10^-4 M = 0.1 mol/m^3" |
| L3 | 1 | `_bv_common.py:275-288` | Missing explicit `kappa_scale_m_s` triggers spurious warnings |
| L4 | 2 | `FluxCurve/bv_observables.py:99-115` | BV gradient array silently truncates (Robin version validates) |
| L5 | 2 | `FluxCurve/bv_run/optimization.py:401-408` | Regularization div-by-zero risk for very small k0 (non-log mode only) |
| L6 | 3 | `Surrogate/nn_model.py:555-579` | Cached normalizer tensors not invalidated on model refit |
| L7 | 3 | `Surrogate/gp_model.py:693-720` | GP gradient uses float32 (NN uses float64 for autograd) |
| L8 | 3 | `Surrogate/objectives.py:246-263` | `_n_evals` double-counts on FD path (bookkeeping only) |
| L9 | 3 | `Surrogate/gp_model.py:742` | `training_bounds` property has no setter (incompatible with io.py) |
| L10 | 3 | `Surrogate/pce_model.py:157-188` | `_transformed_bounds` computed but never used (dead code) |
| L11 | 4 | `Forward/dirichlet_solver.py:169-171` | BCs only on markers 1,3; phi has no opposite-boundary ground |
| L12 | 4 | `Forward/bv_solver/solvers.py:528-532` | Charge continuation doesn't restore z_consts on failure |
| L13 | 4 | `Forward/params.py:71` | z_vals elements not float-normalized in `__post_init__` |
| L14 | 4 | Multiple | Full options dict (including bv_bc) passed as PETSc params |
| L15 | 5 | `Surrogate/nn_training.py:441-446` | `training_bounds` computed on post-split data |
| L16 | 5 | `Surrogate/ismo.py:479-494` | Uncertainty acquisition loops 5000x single-point (slow for GP) |
| L17 | 6 | `Infer_BVMaster_v13_ultimate.py:1160` | `pde_time` can be negative with `--no-pde` |
| L18 | 6 | `Infer_BVMaster_v13_ultimate.py:922` | `surrogate_time` inflated when `--compare` is used |
| L19 | 5 | `Surrogate/ismo.py:864-871` | O(N*M) dedup loop (fine for current sizes) |

---

## Previously Reported Bugs — Status Summary

| Status | Count | Notes |
|--------|-------|-------|
| **FIXED** | ~25 | PTC dt_const, Robin z_vals, conv_cfg defaults, voltage grid validation, PDE failure tracking, ISMO imports, normalizer correction, deepcopy, smoothness penalty, ensemble ddof, load_surrogate types, etc. |
| **STILL OPEN** | ~12 | Mostly LOW/MEDIUM: bv_cfg mutations, c0 vs c_ref BC, Debye length sqrt(2), Nondim compat 2-species hardcoding, noise NaN handling, etc. |
| **DOWNGRADED** | 1 | R-space volume integral: CRITICAL -> MEDIUM (mesh volume = 1.0) |

---

## Recommended Fix Priority

### Immediate (before next pipeline run):
1. **H1+H2**: Fix `test_pipeline_reproducibility.py` imports and voltage grid
2. **H3**: Initialize `last_reason` before attempt loop
3. **M11**: Change RuntimeError to return None in multi-seed runner

### Before next publication/release:
4. **H4**: Fix ISMO `_retrain_surrogate` data leakage
5. **M1**: Replace R-space assembly with `fd.Constant` (latent correctness bug)
6. **M4**: Fix PCE `predict_batch` shape for single-sample
7. **M7**: Fix PCE dispatch in `retrain_surrogate_full`
8. **M14**: Fix `p.get("bv_bc", {})` -> `p.setdefault("bv_bc", {})`

### When touching adjacent code:
9. **M2**: Decouple fail-gradient magnitude from penalty
10. **M5**: Make NN `predict_torch` exception-safe
11. **M6**: Shuffle data before k-fold CV in POD-RBF
12. **M8+M9**: Unify NRMSE normalization convention
13. **M13**: Align PDE phase bounds with surrogate training bounds
