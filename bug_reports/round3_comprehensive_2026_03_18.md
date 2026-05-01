# Comprehensive Bug Audit -- Round 3

**Date:** 2026-03-18
**Method:** 12 parallel specialized agents (9 completed, 3 hit rate limits)
**Focus:** Data flow fidelity, math-vs-code consistency, stale state, sign conventions

---

## Executive Summary

This round focused on **novel angles** not covered by the round-1/round-2 audits: exact math-vs-LaTeX tracing, dimensional analysis, sign convention pipeline tracing, stale state / pyadjoint tape bugs, and serialization roundtrip fidelity.

| Severity | New Findings | Previously Known |
|----------|-------------|-----------------|
| CRITICAL | 2 | 1 (confirmed) |
| HIGH     | 8 | 3 (confirmed) |
| MEDIUM   | 12 | — |
| LOW      | 15 | — |
| **Total**| **37** | **4 confirmed** |

---

## CRITICAL Findings

### CRIT-1 (NEW): Silent non-convergence in all three basic solver loops
**Files:** `Forward/bv_solver/solvers.py:65`, `Forward/robin_solver.py:288`, `Forward/dirichlet_solver.py:248`
**Agent:** solver-convergence

Firedrake's `NonlinearVariationalSolver.solve()` only raises `ConvergenceError` if `snes_error_if_not_converged=True` in PETSc options. The default SNES options in `scripts/_bv_common.py:176-190` do NOT set this flag. All three basic `forsolve*` loops call `solver.solve()` with no convergence check and no try/except. If Newton hits `snes_max_it=200` without converging, **the non-converged solution is silently accepted** and propagated to subsequent timesteps and observables.

The flag IS set in `Inverse/inference_runner/recovery.py:240` and `FluxCurve/recovery.py:62`, so inference paths are protected. But direct `forsolve_bv()` / `forsolve()` callers (training data generation, standalone verification scripts) are NOT.

### CRIT-2 (NEW): Inconsistent pyadjoint tape annotation between sequential and parallel BV paths
**Files:** `FluxCurve/bv_point_solve/__init__.py:611-612` vs `FluxCurve/bv_parallel.py:438`
**Agent:** stale-state

Sequential path wraps `U_prev.assign(U)` in `adj.stop_annotating()`, hiding timestep updates from the adjoint tape. The parallel worker does the same assign WITHOUT `stop_annotating`, meaning the tape records all timestep transitions. The two paths may compute **different adjoint gradients** for the same forward problem. If gradient-based optimization switches between sequential and parallel evaluation, gradient inconsistency could cause optimizer divergence or wrong parameter recovery.

### CRIT-1 (CONFIRMED): Target assembled via `dx` on non-unit domains
**File:** `FluxCurve/bv_point_solve/__init__.py:663-664`
**Agent:** fluxcurve-pipeline

Previously reported. `fd.assemble(target_ctrl * dx)` = `target * Volume(mesh)`. On the current unit-square mesh, Volume=1.0, so results are numerically correct. But the code path is semantically wrong (volume integral of a constant vs. the intended scalar value), and would produce systematically wrong objectives on any non-unit domain.

---

## HIGH Findings

### HIGH-1 (NEW): Overpotential eta omits solution potential phi_s
**File:** `Forward/bv_solver/forms.py:195-198`
**Agent:** math-vs-code

The LaTeX document defines `eta = phi_m - phi_s - E_eq` (line 92). The code computes:
- `use_eta_in_bv=True`: `eta = phi_applied - E_eq` (drops phi_s entirely)
- `use_eta_in_bv=False`: `eta = phi - E_eq` (uses interior field, conflates phi_m - phi_s with phi_s)

**Neither path matches the three-term LaTeX formula.** This is a deliberate modeling simplification (documented in docstring line 69) for Jacobian stability, but it is a mathematical departure from the stated formulation. For thin double layers where the potential drop occurs mostly across the EDL, this approximation is reasonable, but it should be explicitly noted in the V&V report.

### HIGH-2 (NEW): MMS manufactured solutions differ from LaTeX documentation
**File:** `scripts/verification/mms_bv_4species.py` vs `docs/PNP Equation Formulations.tex:384-386`
**Agent:** math-vs-code

The LaTeX describes MMS solutions `c_i = c_0 + A sin(pi*x) sin(pi*y) e^{-t}` (time-dependent, homogeneous Dirichlet). The actual code uses `c_i = c0_i + A_i cos(pi*x) (1-exp(-beta*y))` (steady-state, non-homogeneous). The MMS procedure is still valid, but the LaTeX does not document the test actually implemented.

### HIGH-3 (NEW): Ghost test references -- deleted tests never replaced
**File:** `tests/test_v13_verification.py`
**Agent:** test-audit

`test_v13_verification.py` references `test_inverse_verification.py` and `test_surrogate_fidelity.py` as having "subsumed" 4 of its 7 original tests (zero-noise parameter recovery, PDE roundtrip, surrogate-vs-PDE consistency, multistart basin). **Neither file exists in `tests/`.** These critical verification tests are simply gone.

### HIGH-4 (NEW): No correctness check in pipeline reproducibility tests
**File:** `tests/test_pipeline_reproducibility.py`
**Agent:** test-audit

This test verifies that repeated runs produce the same numbers (regression test), but never checks that inferred parameters are close to true values. It would pass if the pipeline consistently returns wrong answers.

### HIGH-5 (NEW): Entire Inverse package has near-zero test coverage
**Agent:** test-audit

`Inverse/objectives.py`, `Inverse/solver_interface.py`, `Inverse/inference_runner/objective.py`, `Inverse/inference_runner/recovery.py` — the core inference logic — have no dedicated tests.

### HIGH-6 (NEW): Module-level caches keyed by index, not parameters
**File:** `FluxCurve/bv_point_solve/cache.py:11-18`
**Agent:** stale-state

`_all_points_cache` and `_cross_eval_cache` are keyed by voltage point index only. In multi-start optimization, cached solutions from a previous starting point's final iteration persist and warm-start the next starting point at completely different parameter values. While the forward solve re-converges, the stale IC could lead to different convergence paths affecting the adjoint gradient.

### HIGH-7 (NEW): Pickle fragility for RBF/POD-RBF/PCE models
**Files:** `Surrogate/io.py:44`, `Surrogate/pce_model.py:819`
**Agent:** data-io

Scipy `RBFInterpolator` and ChaosPy polynomial objects are pickled whole. Version upgrades to scipy or chaospy can silently break deserialization or produce wrong results.

### HIGH-8 (NEW): No observable-scale normalization in multi-observable fitting
**File:** `Surrogate/objectives.py`
**Agent:** inverse-solver

Neither PDE-based nor surrogate-based objectives normalize observables to comparable scales. If current density is O(1) and peroxide current is O(0.01), the peroxide term contributes ~10,000x less. `secondary_weight` defaults to 1.0 with no automatic scaling.

---

## MEDIUM Findings

| # | Finding | File | Agent |
|---|---------|------|-------|
| M1 | Debye length in `transform.py` uses `potential_scale` (overridable) vs `scales.py` which uses `thermal_voltage_v`. When `potential_scale != V_T`, reported Debye length is wrong. | `Nondim/transform.py:405-408` | nondim-units |
| M2 | PTC silently abandons on Newton failure (breaks out of inner loop without raising or flagging). | `Forward/bv_solver/solvers.py:348-354` | solver-convergence |
| M3 | Charge continuation Phase 1 (neutral sweep) has zero error handling. Newton failure is completely unhandled. | `Forward/bv_solver/solvers.py:482-483` | solver-convergence |
| M4 | Exponent clip value of 50 still allows `exp(50) ~ 5e21`, causing extreme BV flux and Jacobian conditioning issues. | `Forward/bv_solver/forms.py:203-207` | solver-convergence |
| M5 | Robin solver crashes on 1D meshes — `x, y = fd.SpatialCoordinate(mesh)` is hardcoded for 2D. | `Forward/robin_solver.py:239` | solver-convergence |
| M6 | No NaN/Inf check after any solver step. Non-finite values propagate silently. | All solver loops | solver-convergence |
| M7 | CG elements allow negative concentrations in the interior. Only the BV boundary term is regularized, not the bulk transport. | `Forward/bv_solver/forms.py:241-255` | solver-convergence |
| M8 | `SurrogateObjective.__init__` does not validate that target length matches surrogate's `n_eta`. | `Surrogate/objectives.py` | inverse-solver |
| M9 | No early-exit in multistart if all grid objectives are infinite (degenerate surrogate). | `Surrogate/multistart.py` | inverse-solver |
| M10 | float32/float64 asymmetry: NN `predict_batch()` returns float32-precision results dressed as float64, while `predict_torch(requires_grad=True)` returns true float64. | `Surrogate/nn_model.py:509-516` vs `:582-588` | data-io |
| M11 | Train/test split indices have no bounds validation after merge — could reference out-of-bounds rows. | `Surrogate/ismo_retrain.py:245-288` | data-io |
| M12 | If all FluxCurve evaluations have failures, `best_sim_flux` remains all-NaN and is written to output CSV. | `FluxCurve/bv_run/optimization.py:203` | fluxcurve-pipeline |

---

## LOW Findings

| # | Finding | File | Agent |
|---|---------|------|-------|
| L1 | Robin solver `dt_m` is Python float, not `fd.Constant` — prevents adaptive timestepping if ever needed. | `Forward/robin_solver.py:142` | stale-state |
| L2 | `electromigration_prefactor` baked as Python float — not differentiable for temperature/length inference. | `Forward/robin_solver.py:141`, `Forward/bv_solver/forms.py:176` | stale-state |
| L3 | Steric `a` params are `fd.Constant` in robin but `fd.Function(R_space)` in BV solver — robin can't support steric inference. | `Forward/robin_solver.py:150-156` | stale-state |
| L4 | `predict_torch` float64/float32 in-place model conversion not thread-safe. | `Surrogate/nn_model.py:584-587` | surrogate-fidelity |
| L5 | FD gradient step `h=1e-5` uniform for log10(k0) and alpha; near float-point noise floor for flat objectives. | `Surrogate/objectives.py` | inverse-solver |
| L6 | `SurrogateObjective` stores `bounds` attribute but never passes it to optimizer. | `Surrogate/objectives.py:104` | inverse-solver |
| L7 | `_n_evals` counter inflated in FD gradient path. | `Surrogate/objectives.py` | inverse-solver |
| L8 | MMS convergence rate test has no upper bound — superconvergent/erroneous slopes pass silently. | `tests/test_mms_convergence.py` | test-audit |
| L9 | GCI test accepts any non-negative finite value (GCI=500% would pass). | `tests/test_mms_convergence.py` | test-audit |
| L10 | `allow_pickle=True` in NN/GP metadata load opens security surface. | `Surrogate/nn_model.py:690`, `Surrogate/gp_model.py:839` | data-io |
| L11 | Parameter column order in training .npz is by convention, not metadata. | `scripts/surrogate/generate_training_data.py` | data-io |
| L12 | IntervalMesh electrode marker mismatch: config defaults to 3 (RectangleMesh bottom), but IntervalMesh uses 1 (left). Must be manually overridden. | `Forward/bv_solver/config.py:32` | fluxcurve-pipeline |
| L13 | BV target generation uses `i_scale=-current_density_scale` (double-negation pattern) — fragile for new callers. | `FluxCurve/bv_run/io.py:76` | sign-conventions |
| L14 | Robin pipeline applies `observable_scale` to loaded target; BV pipeline does not. | `FluxCurve/run.py:522` vs `FluxCurve/bv_run/io.py:49` | fluxcurve-pipeline |
| L15 | Backward compatibility patches in `load_surrogate` set `None` defaults that could mask missing attributes. | `Surrogate/io.py:74-84` | data-io |

---

## Clean Findings (Verified Correct)

The following areas were verified and found to be **correctly implemented**:

- **Nernst-Planck weak form** (drift, steric terms) matches LaTeX in both dimensional and nondim modes
- **Poisson weak form** coefficients (eps, charge_rhs) correct in both modes
- **BV exponent scale**: F/(RT) dimensional, 1.0 nondim — correct
- **Stoichiometric sign convention**: internally consistent (cathodic consumption / anodic production)
- **All sign conventions** traced end-to-end: no hard sign errors found
- **Thermal voltage** V_T = RT/F: correct in both `scales.py` and `transform.py`
- **Kappa scale** D_ref/L: consistent everywhere
- **Current density scale** F*D*c/L: consistent between `scales.py` and `transform.py`
- **All `*_inputs_are_dimensionless` flags** correctly skip corresponding divisions
- **NN ensemble**: normalizers saved/loaded correctly, train/eval mode handled (LayerNorm, no dropout)
- **POD-RBF**: SVD, truncation, log-transform all correct
- **PCE**: polynomial basis, Sobol indices, analytic gradient all correct
- **Cascade model**: pass ordering, subset objectives, final loss evaluation all correct
- **Parallel FluxCurve**: no shared mutable state, results correctly indexed
- **Failed points**: do not poison predictor/continuation chain
- **CSV I/O**: header-based reading prevents column misalignment
- **Ensemble grid validation**: members checked for `phi_applied` consistency

---

## Recommended Fix Priority

### Immediate (blocks correctness)
1. **CRIT-1**: Add `snes_error_if_not_converged: True` to default SNES options, or add explicit convergence checks in `forsolve_bv/forsolve`
2. **CRIT-2**: Standardize pyadjoint tape annotation for `U_prev.assign(U)` between sequential and parallel paths
3. **HIGH-3**: Create the missing `test_inverse_verification.py` and `test_surrogate_fidelity.py` test files

### Soon (impacts reliability)
4. **HIGH-1**: Document the eta simplification in V&V report; consider implementing the full `phi_m - phi_s - E_eq`
5. **HIGH-6**: Add parameter-hash invalidation to module-level caches, or clear caches between multi-start restarts
6. **HIGH-8**: Add automatic observable-scale normalization to multi-observable objectives
7. **M6**: Add NaN/Inf checks after each solver step

### When convenient
8. **HIGH-7**: Migrate RBF/PCE serialization from pickle to explicit numpy-based format
9. **M1**: Use `thermal_voltage_v` (not `potential_scale`) for Debye length in `transform.py`
10. **M5**: Fix robin solver for 1D meshes

---

## Comparison with Round 1/2 Audit

| Category | Round 1/2 (2026-03-17) | Round 3 (2026-03-18) | Status |
|----------|----------------------|---------------------|--------|
| Total bugs | 88 | 37 new + 4 confirmed | Complementary |
| CRITICAL | 3 | 2 new + 1 confirmed | CRIT-1 (volume bug) confirmed; 2 new CRITs found |
| Math-vs-LaTeX | Not audited | Audited term-by-term | **New coverage** |
| Sign conventions | Partially checked | Full pipeline trace | **All clear** |
| Dimensional analysis | Not audited | Full unit tracking | **1 inconsistency found** |
| Stale state / pyadjoint | Not audited | Comprehensive | **2 HIGH bugs found** |
| Test correctness | Tolerance issues flagged | Ghost references + no correctness checks found | **3 HIGH findings** |
