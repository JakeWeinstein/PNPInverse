# Codebase Concerns

**Analysis Date:** 2026-03-06

## Tech Debt

**Massive Script Version Sprawl (inference scripts):**
- Issue: 42 inference scripts totaling 11,245 lines in `scripts/inference/`, many are incremental versions of the same workflow (v2 through v7 of `Infer_BVMaster_charged_*.py`). Similarly, `scripts/surrogate/` has v8 through v13 versioned scripts (7 versioned files). These are effectively frozen snapshots of previous experiments, not maintained code.
- Files: `scripts/inference/Infer_BVMaster_charged.py`, `scripts/inference/Infer_BVMaster_charged_v2.py`, `scripts/inference/Infer_BVMaster_charged_v3.py`, `scripts/inference/Infer_BVMaster_charged_v4.py`, `scripts/inference/Infer_BVMaster_charged_v5.py`, `scripts/inference/Infer_BVMaster_charged_v6.py`, `scripts/inference/Infer_BVMaster_charged_v7.py`, `scripts/surrogate/Infer_BVMaster_charged_v8_surrogate.py`, `scripts/surrogate/Infer_BVMaster_charged_v8_1_surrogate.py`, `scripts/surrogate/Infer_BVMaster_charged_v9_surrogate.py`, `scripts/surrogate/Infer_BVMaster_charged_v10_fixed_pde.py`, `scripts/surrogate/Infer_BVMaster_charged_v11_surrogate_pde.py`, `scripts/surrogate/Infer_BVMaster_charged_v12_nn_surrogate_pde.py`, `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py`
- Impact: Maintenance burden; bug fixes in one version are not propagated to others. Confusing for new contributors to identify which script is canonical. v13 is the current "ultimate" version.
- Fix approach: Archive old versions into a `scripts/archive/` directory. Keep only v13 as the active inference pipeline. If old experiments need reproducing, use git history.

**Pervasive `sys.path.insert(0, _ROOT)` Hack:**
- Issue: Every script and test file manually computes the project root and inserts it into `sys.path`. This is done in 90+ locations across the codebase. The project has a `pyproject.toml` with proper package discovery but scripts are not using it via an editable install.
- Files: `scripts/_bv_common.py` (lines 20-21, 59-61), `tests/conftest.py` (lines 18-21), and 88+ other scripts/tests
- Impact: Fragile import resolution; path computation breaks if directory structure changes. Makes it impossible to run scripts from arbitrary working directories reliably.
- Fix approach: Use `pip install -e .` (editable install) for the project. Remove all `sys.path.insert` hacks. Add a `scripts` package or use proper entry points.

**Disabled Replay Mode:**
- Issue: The replay mode optimization (reusing per-phi reduced functionals for fast re-evaluation) is permanently disabled with `replay_mode_enabled: bool = False` and a comment "TEMPORARILY DISABLED: replay can produce invalid/non-steady evaluations." This has been disabled for an indeterminate period.
- Files: `FluxCurve/config.py` (line 83-84), `FluxCurve/run.py` (line 502), `FluxCurve/replay.py`
- Impact: Dead code path (~500 lines in `FluxCurve/replay.py`). Performance optimization that could reduce PDE evaluation time is unavailable.
- Fix approach: Either fix the validity issue and re-enable, or remove the replay infrastructure entirely to reduce complexity.

**Global Mutable State in Module-Level Caches:**
- Issue: Multiple modules use `global` variables for caches and worker state. These are mutated via setter functions and create hidden coupling between callers.
- Files: `FluxCurve/bv_point_solve/cache.py` (lines 11-24: `_cross_eval_cache`, `_all_points_cache`, `_cache_populated`, `_parallel_pool`), `FluxCurve/bv_parallel.py` (lines 102, 147, 298: `_WORKER_CONFIG`, `_WORKER_MESH`), `FluxCurve/point_solve.py` (line 30: `_PARALLEL_POINT_CONFIG`), `Surrogate/training.py` (line 582: `_TRAIN_WORKER_STATE`)
- Impact: Cache state leaks between test runs if not explicitly cleared. Makes concurrent usage impossible. Difficult to reason about state during debugging.
- Fix approach: Encapsulate caches in a context manager or class instance that can be scoped per-run. Pass state explicitly rather than using module globals.

**Oversized Files:**
- Issue: Several core files significantly exceed 800-line guidelines, making them hard to navigate and maintain.
- Files: `FluxCurve/bv_run/pipelines.py` (1,829 lines), `scripts/surrogate/cascade_pde_hybrid.py` (1,675 lines), `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` (1,257 lines), `Surrogate/training.py` (938 lines), `FluxCurve/bv_run/optimization.py` (879 lines), `FluxCurve/bv_point_solve/__init__.py` (774 lines), `FluxCurve/bv_parallel.py` (766 lines)
- Impact: Difficult to understand, test, or modify individual concerns. High cognitive load for maintenance.
- Fix approach: Extract logical sub-modules. For example, split `pipelines.py` by pipeline type (single-k0, multi-observable, multi-pH). Move helper functions in `__init__.py` to dedicated submodules.

## Known Bugs

**KMP_DUPLICATE_LIB_OK Environment Workaround:**
- Symptoms: Firedrake + PyTorch coexistence causes OpenMP duplicate library errors. Suppressed via `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`.
- Files: `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` (line 40), `scripts/surrogate/Infer_BVMaster_charged_v12_nn_surrogate_pde.py` (line 36), `scripts/surrogate/overnight_train_v11.py` (line 26), `tests/test_v13_verification.py` (line 35)
- Trigger: Running any script that imports both Firedrake and PyTorch
- Workaround: Environment variable suppresses the error but does not fix the underlying library conflict. Could mask real threading issues.

## Security Considerations

**No Sensitive Data Detected:**
- Risk: Low. This is a scientific computing codebase with no web endpoints, user authentication, or external API keys.
- Files: No `.env` files detected. No credentials or API keys in code.
- Current mitigation: N/A
- Recommendations: If the codebase ever grows to include remote computation or cloud resources, add proper secret management.

## Performance Bottlenecks

**PDE Forward Solves (Primary Bottleneck):**
- Problem: Each PDE evaluation (Firedrake SNES solve) takes 1-5 seconds per voltage point. A full I-V curve sweep with 15-30 points takes 80-300 seconds per L-BFGS-B iteration.
- Files: `FluxCurve/bv_point_solve/__init__.py`, `Forward/steady_state/bv.py`, `FluxCurve/bv_parallel.py`
- Cause: Nonlinear PDE (Poisson-Nernst-Planck + Butler-Volmer) requires iterative Newton solves. Each optimization iteration requires a full voltage sweep.
- Improvement path: The surrogate model approach (v9-v13) already addresses this by using RBF/NN surrogates for initial warm-starting. The disabled replay mode could further help if fixed. Parallel point solves (`bv_parallel.py`) provide additional speedup when Firedrake supports multi-process.

**Finite-Difference Gradients for Surrogate Objectives:**
- Problem: Surrogate gradient computation uses central finite differences (8 evaluations for 4 parameters) rather than analytical gradients.
- Files: `Surrogate/objectives.py` (line 5: "Gradients are computed via central finite differences")
- Cause: RBF surrogate does not provide analytical derivatives. NN surrogate could use autograd but currently does not.
- Improvement path: Implement analytical gradients for the NN ensemble surrogate via PyTorch autograd. This would halve the surrogate objective evaluation cost.

## Fragile Areas

**BV Point Solver Voltage Continuation:**
- Files: `FluxCurve/bv_point_solve/__init__.py`, `FluxCurve/bv_point_solve/forward.py`, `FluxCurve/bv_point_solve/predictor.py`, `FluxCurve/bv_point_solve/cache.py`
- Why fragile: The sequential warm-start sweep (solving from small to large |eta|) is critical for convergence at high overpotentials. Changing the voltage ordering, mesh parameters, or SNES options can cause cascading divergence. The predictor step (quadratic/linear hybrid) and bridge point logic add complexity.
- Safe modification: Always test changes with both shallow (cathodic-only) and full (symmetric) voltage ranges. Verify convergence at the most extreme voltage points. Never change the sweep ordering without understanding the continuation strategy.
- Test coverage: `tests/test_v13_verification.py` covers the end-to-end pipeline. No unit tests exist for the individual point solver internals (predictor, bridge points, cache logic).

**Forward Recovery Multi-Stage Fallback:**
- Files: `FluxCurve/config.py` (`ForwardRecoveryConfig`), `FluxCurve/recovery.py`, `FluxCurve/bv_point_solve/forward.py`
- Why fragile: The 3-stage recovery (max-it increase, anisotropy, tolerance relaxation) with 8 total attempts creates a complex state machine. Line search schedules cycle through 4 strategies. Each stage mutates solver options in place.
- Safe modification: Add recovery attempts only at the end of the cascade. Never remove existing stages without verifying convergence on the full voltage range.
- Test coverage: No dedicated unit tests for recovery logic.

**Nondimensionalization Pipeline:**
- Files: `Nondim/transform.py`, `Nondim/scales.py`, `Nondim/constants.py`, `Nondim/compat.py`
- Why fragile: Correct nondimensionalization is mathematically critical. A past bug hardcoded a value to 1.0 that should have been physical (noted at `Nondim/transform.py` line 241). Dimensional analysis errors silently produce wrong results without runtime errors.
- Safe modification: Always run the MMS convergence verification (`scripts/verification/mms_bv_convergence.py`) after any nondim changes. The test suite at `tests/test_nondim.py` provides good coverage for the scaling computations.
- Test coverage: `tests/test_nondim.py` is well-tested (274+ lines). MMS verification scripts exist but are not part of the automated test suite.

## Scaling Limits

**Memory Usage with Large Voltage Grids:**
- Current capacity: 15-30 voltage points per I-V curve
- Limit: Each Firedrake function space and solution vector consumes significant memory. The checkpoint cache (`_all_points_cache`) stores NumPy arrays for every voltage point, growing linearly.
- Scaling path: Use HDF5 checkpointing instead of in-memory caches for large voltage grids. Consider adaptive voltage point selection.

## Dependencies at Risk

**Firedrake (FEM Framework):**
- Risk: Firedrake is installed via its own custom installer (not pip), creating a non-standard dependency. It is not listed in `pyproject.toml` dependencies (only noted in a comment at line 15).
- Impact: The entire Forward solver, PDE-based inference, and adjoint gradient computation depend on Firedrake. All tests marked `slow` require it.
- Migration plan: No alternative FEM framework provides the same automatic adjoint differentiation. Firedrake is a hard dependency.

**PyTorch (NN Surrogate, Optional):**
- Risk: Not declared in `pyproject.toml` dependencies but required for the NN ensemble surrogate (`Surrogate/nn_model.py`, `Surrogate/nn_training.py`, `Surrogate/ensemble.py`). The `KMP_DUPLICATE_LIB_OK` hack is needed for coexistence with Firedrake.
- Impact: NN ensemble surrogate is the recommended model type in v13. Failure to import PyTorch silently falls back to RBF-only.
- Migration plan: Add `torch` as an optional dependency in `pyproject.toml` under an `[nn]` extra.

**No Pinned Dependency Versions:**
- Risk: `pyproject.toml` lists `numpy`, `scipy`, `matplotlib`, `h5py` without version constraints. Future breaking changes in NumPy 2.x or SciPy API changes could break the codebase without warning.
- Impact: Reproducibility of scientific results is at risk.
- Migration plan: Pin minimum versions (e.g., `numpy>=1.24,<3`, `scipy>=1.10`). Consider a `requirements-lock.txt` for exact reproducibility.

## Missing Critical Features

**No Structured Logging:**
- Problem: All diagnostic output uses `print()` statements (140+ across core modules). No structured logging framework.
- Blocks: Cannot filter log levels, redirect output, or aggregate diagnostics. Makes debugging production runs difficult.

**No CI/CD Pipeline:**
- Problem: No GitHub Actions, Jenkins, or other CI configuration detected. Tests must be run manually.
- Blocks: No automated regression detection. No automated validation that changes pass the test suite.

## Test Coverage Gaps

**Forward Solver Internals (Firedrake-dependent):**
- What's not tested: `Forward/bv_solver/forms.py`, `Forward/bv_solver/solvers.py`, `Forward/dirichlet_solver.py`, `Forward/robin_solver.py` have no dedicated unit tests. Coverage depends entirely on end-to-end tests that require Firedrake.
- Files: `Forward/bv_solver/forms.py` (388+ lines), `Forward/bv_solver/solvers.py` (434+ lines), `Forward/dirichlet_solver.py` (224+ lines)
- Risk: Bugs in variational form assembly or solver configuration would not be caught until full pipeline runs. These are the most mathematically sensitive parts of the codebase.
- Priority: Medium (MMS verification scripts partially cover this, but are not automated)

**FluxCurve Point Solver Internals:**
- What's not tested: Predictor step logic, bridge point insertion, SER adaptive timestepping, recovery fallback stages, parallel dispatch logic.
- Files: `FluxCurve/bv_point_solve/predictor.py`, `FluxCurve/bv_point_solve/forward.py`, `FluxCurve/bv_point_solve/cache.py`, `FluxCurve/bv_parallel.py`
- Risk: Complex numerical logic with many edge cases (divergence, cache misses, worker failures). Changes could introduce subtle convergence regressions.
- Priority: High

**Surrogate Training Pipeline:**
- What's not tested: `Surrogate/training.py` (938 lines) and `Surrogate/nn_training.py` have no dedicated tests. Training correctness is validated only by downstream inference accuracy.
- Files: `Surrogate/training.py`, `Surrogate/nn_training.py`
- Risk: Training bugs could produce poor surrogates that silently degrade inference quality.
- Priority: Medium

**Scripts Not Tested:**
- What's not tested: 42 inference scripts and 10+ study scripts have no automated tests. Some are tested indirectly via `tests/test_v13_verification.py` and `tests/test_cascade_pde_hybrid.py`, but most older scripts have zero coverage.
- Files: All files under `scripts/inference/`, `scripts/studies/`
- Risk: Low (these are experimental scripts, not library code). Old scripts may already be broken.
- Priority: Low (focus testing effort on the core library modules instead)

---

*Concerns audit: 2026-03-06*
