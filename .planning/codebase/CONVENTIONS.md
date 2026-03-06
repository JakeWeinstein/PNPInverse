# Coding Conventions

**Analysis Date:** 2026-03-06

## Naming Patterns

**Files:**
- snake_case for all Python modules: `surrogate_model.py`, `solver_interface.py`, `bv_parallel.py`
- Test files: `test_<module_or_feature>.py` in `tests/`
- Leading underscore for internal/shared helpers: `scripts/_bv_common.py`
- Directories use PascalCase for top-level packages: `Forward/`, `Inverse/`, `Surrogate/`, `Nondim/`, `FluxCurve/`
- Directories use snake_case for sub-packages: `bv_point_solve/`, `bv_run/`, `inference_runner/`, `steady_state/`

**Functions:**
- snake_case universally: `run_cascade_inference()`, `compute_bv_current_density()`, `make_bv_solver_params()`
- Private/internal functions prefixed with underscore: `_make_subset_objective_fn()`, `_run_pass1()`, `_clear_caches()`
- Builder/factory functions prefixed with `make_` or `build_`: `make_bv_solver_params()`, `build_physical_scales()`, `build_model_scaling()`
- Computation functions prefixed with `compute_`: `compute_bv_reaction_rates()`, `compute_i_scale()`

**Variables:**
- snake_case for local variables and parameters: `target_cd`, `subset_idx`, `secondary_weight`
- UPPER_SNAKE_CASE for module-level constants: `FARADAY_CONSTANT`, `K0_HAT_R1`, `I_SCALE`, `SNES_OPTS`
- Dimensionless quantities suffixed with `_hat`: `D_O2_HAT`, `K0_HAT_R1`, `C_HP_HAT`
- Physical quantities include unit suffix: `d_species_m2_s`, `thermal_voltage_v`, `current_density_a_m2`, `length_scale_m`
- Boolean flags use descriptive names: `log_space_k0`, `skip_polish`, `blob_initial_condition`

**Types/Classes:**
- PascalCase for all classes: `SolverParams`, `CascadeConfig`, `EnsembleMeanWrapper`, `SteadyStateResult`
- Config dataclasses suffixed with `Config`: `CascadeConfig`, `MultiStartConfig`, `SurrogateConfig`, `SteadyStateConfig`
- Result dataclasses suffixed with `Result`: `CascadeResult`, `MultiStartResult`, `SteadyStateResult`, `CascadePassResult`

## Code Style

**Formatting:**
- No explicit formatter configuration (no `.prettierrc`, `black` config, or `pyproject.toml` formatter section)
- 4-space indentation (standard Python)
- Lines generally kept under ~100 characters, with occasional longer lines for complex expressions

**Linting:**
- No explicit linter configuration detected (no `.flake8`, `ruff.toml`, or `pylintrc`)
- Code follows PEP 8 conventions naturally
- `# noqa: F401` used for intentional unused imports (e.g., `import firedrake  # noqa: F401` for availability checks)

**Type Annotations:**
- Use `from __future__ import annotations` at the top of every module for PEP 604 union syntax (`X | None`)
- Function signatures are type-annotated: parameters and return types
- Use `typing` imports for complex types: `Sequence`, `Tuple`, `Optional`, `Dict`, `List`
- Mix of old-style (`Optional[X]`) and new-style (`X | None`) union syntax -- prefer new-style going forward

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first)
2. Standard library: `os`, `sys`, `time`, `copy`, `math`, `dataclasses`, `typing`, `contextlib`
3. Third-party: `numpy as np`, `scipy`, `pytest`, `matplotlib`
4. Internal packages: `Forward.*`, `Inverse.*`, `Surrogate.*`, `Nondim.*`, `FluxCurve.*`

**Path Aliases:**
- No path aliases configured (no `pyproject.toml` `[tool.setuptools.package-dir]` or similar)
- Tests and scripts manually add the project root to `sys.path`:
  ```python
  _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
  _ROOT = os.path.dirname(_THIS_DIR)
  if _ROOT not in sys.path:
      sys.path.insert(0, _ROOT)
  ```
- Internal imports use absolute paths from package root: `from Forward.params import SolverParams`, `from Surrogate.cascade import CascadeConfig`

**Standard Aliases:**
- `import numpy as np` (always)
- `import firedrake as fd` (in FEM-dependent code)
- `from scipy.optimize import minimize` (direct import)

## Error Handling

**Patterns:**
- `ValueError` for invalid inputs with descriptive `match` strings:
  ```python
  if reaction_index not in (0, 1):
      raise ValueError(f"reaction_index must be 0 or 1, got {reaction_index}")
  ```
- `FileNotFoundError` for missing files/directories:
  ```python
  raise FileNotFoundError(f"Ensemble member not found: {member_path}")
  ```
- Solver failures return result objects with `converged=False` and `failure_reason` string rather than raising exceptions
- Exception handling in FEM solves wraps `solver.solve()` in try/except and returns a failure result:
  ```python
  try:
      solver.solve()
  except Exception as exc:
      return SteadyStateResult(converged=False, failure_reason=f"{type(exc).__name__}: {exc}")
  ```
- Input validation at function entry, fail-fast with `ValueError`

## Logging

**Framework:** `print()` statements (no structured logging framework)

**Patterns:**
- Controlled by `verbose: bool` parameter on config dataclasses
- Prefixed with bracketed tags: `[Cascade]`, `[bv_steady]`, `[params]`
- Progress messages include numeric values with format specifiers: `f"loss={p1.loss:.4e}"`
- Suppress output entirely when `verbose=False` (tested explicitly in test suite)

## Comments

**When to Comment:**
- Module-level docstrings describe purpose, public API, and usage examples
- Section separators use `# ---------------------------------------------------------------------------` horizontal rules
- Inline comments explain physics rationale or numerical choices: `# softplus regularization in BV convergence`
- `# Legacy` prefix flags backward-compatibility code paths

**Docstrings:**
- NumPy-style docstrings with `Parameters`, `Returns`, `Raises`, `Attributes` sections
- Class docstrings include `Attributes` section listing all fields
- Module docstrings list public API classes/functions
- Example usage shown with `::` code blocks in module docstrings

## Function Design

**Size:** Functions are moderate-length (20-80 lines typical). Complex solver functions can reach 100+ lines (e.g., `sweep_phi_applied_steady_bv_flux` in `Forward/steady_state/bv.py`)

**Parameters:**
- Keyword-only arguments enforced with `*` separator for factory functions: `make_bv_solver_params(*, eta_hat, dt, ...)`
- Config dataclasses group related parameters instead of long parameter lists
- Default values provided for optional parameters
- `np.ndarray` inputs wrapped with `np.asarray(x, dtype=float)` at function entry

**Return Values:**
- Frozen dataclass instances for results: `CascadeResult`, `MultiStartResult`, `SteadyStateResult`
- Dict returns for surrogate predictions: `{"current_density": ..., "peroxide_current": ..., "phi_applied": ...}`
- Tuples for multi-value private returns: `(objective_fn, gradient_fn, eval_counter_dict)`

## Module Design

**Exports:**
- Public API listed explicitly in module docstrings
- No `__all__` definitions (not detected in any module)
- Private helpers prefixed with underscore

**Barrel Files:**
- `__init__.py` files exist in packages (`FluxCurve/__init__.py`, `Inverse/__init__.py`, etc.)
- Re-exports from sub-modules where applicable (e.g., `Forward/steady_state/__init__.py`)

## Data Structures

**Frozen Dataclasses (Primary Pattern):**
- All configuration and result types use `@dataclass(frozen=True)`: `CascadeConfig`, `CascadePassResult`, `CascadeResult`, `MultiStartConfig`, `SolverParams`, `SpeciesConfig`
- Immutability enforced -- mutation raises `AttributeError` (verified in tests)
- Copy-then-modify via `dataclasses.replace()` for SolverParams modifications

**Mutable Dataclasses:**
- `SteadyStateConfig` and `SteadyStateResult` use `@dataclass` (not frozen) -- these are simple data containers

**Constants as Module-Level Values:**
- Physical constants in `Nondim/constants.py` and `scripts/_bv_common.py`
- Single source of truth pattern: import from canonical location, do not redefine

## Numerical Conventions

- Log-space for rate constants: `log10(k0)` in optimizer control vectors, `10.0 ** x[0]` for conversion
- Central finite differences for gradients: `(f(x+h) - f(x-h)) / (2h)` with `h = 1e-5`
- NaN masking for missing data: `valid_cd = ~np.isnan(target_cd)`
- Guard against log(0): `np.log10(max(value, 1e-30))`
- `float()` casts for scalar NumPy values returned from optimization

---

*Convention analysis: 2026-03-06*
