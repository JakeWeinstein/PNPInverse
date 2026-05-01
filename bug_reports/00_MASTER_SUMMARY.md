# Comprehensive Bug Report -- Master Summary

**Date:** 2026-03-17
**Scope:** Full PNPInverse repository
**Method:** 15 parallel bug-checker agents across 4 focus areas

---

## Top-Line Stats

| Severity | Count |
|----------|-------|
| CRITICAL | 3     |
| HIGH     | 20    |
| MEDIUM   | 30    |
| LOW      | 35    |
| **Total** | **88** |

---

## CRITICAL Bugs (Fix Immediately)

### 1. R-space target volume integration -- wrong parameter inference
**Report:** [04_fluxcurve_pipeline_numerics.md](04_fluxcurve_pipeline_numerics.md) Bug 1
**Files:** `FluxCurve/observables.py:101-113`, `point_solve.py:294`, `bv_point_solve/forward.py:276`, `bv_parallel.py:241`
**Impact:** `fd.assemble(target_ctrl * dx)` evaluates to `target * Volume(mesh)` instead of `target`. If mesh volume != 1, the optimizer minimizes the wrong objective and all inferred parameters are systematically biased. Affects both Robin and BV pipelines.

### 2. Bogus voltage grid validation (always True)
**Report:** [15_ismo_module_correctness.md](15_ismo_module_correctness.md) Bug 1
**File:** `Surrogate/ismo_pde_eval.py:532`
**Impact:** Due to Python operator precedence, the validation is a no-op. Voltage grid mismatches go undetected during ISMO data integration, enabling silent data corruption.

### 3. Test suite: 100% error tolerance in parameter recovery
**Report:** [12_test_coverage_gaps.md](12_test_coverage_gaps.md) Bug 1
**File:** `tests/test_multistart.py:332`
**Impact:** `max_err < 1.0` means parameters can be 2x their true value and still pass. Masks real regression bugs.

---

## HIGH Bugs (Fix Soon) -- Top 10

| # | Description | Report | File |
|---|-------------|--------|------|
| 1 | PTC solver never assigns adapted dt to weak form -- PTC is non-functional | [01](01_forward_solver_numerics.md) | `bv_solver/solvers.py:297-369` |
| 2 | `configure_bv_solver_params` loses k0/alpha mutations when bv_bc key missing | [06](06_forward_inverse_error_handling.md) | `steady_state/bv.py:50-66` |
| 3 | `conv_cfg` KeyError when solver_params[10] is not a dict | [06](06_forward_inverse_error_handling.md) | `bv_solver/forms.py:190-231` |
| 4 | NaN predictions silently zeroed in multistart grid evaluation | [03](03_inverse_solver_numerics.md) | `multistart.py:253-254` |
| 5 | GP `predict_torch` API mismatch with NN -- wrong autograd gradients | [05](05_surrogate_model_logic.md) | `gp_model.py:615-649` |
| 6 | Test set used as validation during ISMO retraining (data leakage) | [05](05_surrogate_model_logic.md) | `ismo_retrain.py:596-606` |
| 7 | PDE failure breaks parameter-curve correspondence in ISMO runner | [15](15_ismo_module_correctness.md) | `run_ismo.py:508-537` |
| 8 | Wrong module names in ISMO runner imports (always falls back to LHS) | [15](15_ismo_module_correctness.md) | `run_ismo.py:73-84` |
| 9 | Wrong sign on observable_scale flips I-V polarity | [07](07_scripts_logic_errors.md) | `sensitivity_visualization.py:318` |
| 10 | Failed FluxCurve points: huge penalty but zero gradient | [04](04_fluxcurve_pipeline_numerics.md) | `curve_eval.py:61-67` |

---

## Reports Index

| # | Report | Focus | CRIT | HIGH | MED | LOW |
|---|--------|-------|------|------|-----|-----|
| 01 | [Forward Solver Numerics](01_forward_solver_numerics.md) | Scientific correctness | 0 | 1 | 2 | 4 |
| 02 | [Nondim Package](02_nondim_package_correctness.md) | Scientific correctness | 0 | 0 | 2 | 3 |
| 03 | [Inverse Solver Numerics](03_inverse_solver_numerics.md) | Scientific correctness | 0 | 1 | 3 | 4 |
| 04 | [FluxCurve Pipeline](04_fluxcurve_pipeline_numerics.md) | Scientific correctness | 1 | 4 | 2 | 0 |
| 05 | [Surrogate Model Logic](05_surrogate_model_logic.md) | Code quality / Logic | 0 | 4 | 6 | 2 |
| 06 | [Forward/Inverse Error Handling](06_forward_inverse_error_handling.md) | Code quality / Logic | 0 | 2 | 5 | 3 |
| 07 | [Scripts Logic Errors](07_scripts_logic_errors.md) | Code quality / Logic | 0 | 2 | 7 | 5 |
| 08 | [Surrogate Code Quality](08_surrogate_code_quality.md) | Code quality / Logic | 0 | 2 | 4 | 6 |
| 09 | [Data/IO File Paths](09_data_io_file_paths.md) | Data / IO | 0 | 2 | 6 | 2 |
| 10 | [Config Consistency](10_config_consistency.md) | Data / IO | 0 | 0 | 3 | 4 |
| 11 | [Surrogate Data Pipeline](11_surrogate_data_pipeline.md) | Data / IO | 0 | 1 | 3 | 5 |
| 12 | [Test Coverage Gaps](12_test_coverage_gaps.md) | Testing | 1 | 6 | 4 | 4 |
| 13 | [Test Correctness](13_test_correctness.md) | Testing | 0 | 0 | 3 | 4 |
| 14 | [Writeup/LaTeX Consistency](14_writeup_latex_consistency.md) | Documentation | 0 | 2 | 5 | 2 |
| 15 | [ISMO Module](15_ismo_module_correctness.md) | Cross-cutting | 1 | 7 | 4 | 1 |

---

## Coverage Gaps (No Tests At All)

- **FluxCurve/** -- 18 modules, zero direct tests (highest risk)
- **Inverse/** -- 7 of 8 modules untested
- **Forward/** -- Core PDE solvers (`dirichlet_solver`, `robin_solver`, `bv_solver/forms`) untested
- **Surrogate/** -- `cascade.py`, `ismo.py`, `gp_model.py`, `pce_model.py` untested

---

## Recommended Fix Order

1. **Verify mesh volume** -- If != 1.0, Bug CRIT-1 is actively producing wrong inference results
2. **Fix ISMO runner imports** -- Currently non-functional for real acquisition/retrain
3. **Fix PTC solver** -- Add `dt_const.assign(dt_ptc)` to enable actual pseudo-transient continuation
4. **Fix bv_solver config** -- `setdefault` instead of `get` for bv_bc dict
5. **Fix NaN handling** in multistart grid evaluation
6. **Tighten test tolerances** -- 100% error is unacceptable for parameter recovery tests
7. **Fix V&V report** -- Wrong test set size (479 vs 524), BV equation doesn't match code
