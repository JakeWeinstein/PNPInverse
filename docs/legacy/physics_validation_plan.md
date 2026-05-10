# Physics Validation — Per-File Implementation Plan

**Date**: 2026-04-13
**Companion**: `docs/physics_validation_log.md` (motivation, classification, exploration results)

Each section below is an independent unit of work that can be implemented in parallel.
Checks reference IDs from the classification table (F1-F7 = failures, W1-W11 = warnings).

---

## File 0: `Forward/bv_solver/validation.py` (NEW FILE)

**Purpose**: Shared validation module. All other files import from here.

### What to create:

```python
# Forward/bv_solver/validation.py
"""Physics validation for PNP-BV solver solutions.

Two levels:
- FAIL: data point is discarded, function raises or returns a flag
- WARN: logged via warnings.warn(), data point is kept
"""

@dataclass(frozen=True)
class ValidationResult:
    valid: bool                    # False if any FAIL triggered
    failures: list[str]            # e.g. ["F1: negative c_O2, min=-0.42"]
    warnings: list[str]            # e.g. ["W1: clip saturated at V=0.3"]

def validate_solution_state(U, *, n_species, c_bulk, phi_applied,
                            z_vals, eps_c, exponent_clip,
                            species_names=None) -> ValidationResult:
    """Check F1, F4, F6, W2, W3, W5, W6, W11 on a Firedrake Function U.

    Parameters
    ----------
    U : firedrake.Function
        Mixed function [c_0, ..., c_{n-1}, phi].
    n_species : int
    c_bulk : list[float]
        Bulk (initial/boundary) concentration per species.
    phi_applied : float
        Applied potential (nondimensional).
    z_vals : list[int]
        Charge numbers per species.
    eps_c : float
        Concentration floor from config.
    exponent_clip : float
        Exponent clip value from config.
    species_names : list[str] | None
        For readable messages, e.g. ["O2", "H2O2", "H+", "ClO4-"].
    """
    # F1: negative concentration — any c_i < 0
    # F4: concentration floor domination — min(c_i at electrode boundary) <= eps_c * 1.1
    # F6: H2O2 > c_O2_bulk (stoichiometric limit, species indices by convention)
    # W2: c_i > c_bulk_i * 1.05 in top half of domain
    # W3: phi outside [min(phi_applied,0), max(phi_applied,0)] * 1.1
    # W5: H+ drops >3 orders below bulk outside Debye layer
    # W6: non-monotonic neutral species profiles (grad sign changes)
    # W11: Gibbs oscillations in first 10 elements from electrode

def validate_observables(cd, pc, *, I_lim, phi_applied, V_T) -> ValidationResult:
    """Check F2, F3, F7 on assembled observables.

    Parameters
    ----------
    cd : float
        Current density (already scaled to mA/cm^2 or nondim).
    pc : float
        Peroxide current (same units).
    I_lim : float
        Diffusion-limited current (same units). Compute as n*F*D*c_bulk/L.
    phi_applied : float
        Applied potential (nondim).
    V_T : float
        Thermal voltage for determining cathodic/anodic regime.
    """
    # F2: |cd| > |I_lim| * 1.05  (5% tolerance for numerics)
    # F3: |pc/cd| > 1.05  (selectivity > 100%)
    # F7: pc > 0 at cathodic overpotentials (wrong sign)

def validate_steady_state(flux_history, *, bulk_integral_history=None) -> ValidationResult:
    """Check W8 on steady-state convergence quality.

    Parameters
    ----------
    flux_history : list[float]
        Last N flux values from the time-stepping loop.
    bulk_integral_history : list[float] | None
        Last N integral(c_i) values (optional, for false-SS detection).
    """
    # W8: flux converged but bulk integral still drifting >0.1%

def check_clip_saturation(eta_raw, *, exponent_clip, bv_exp_scale,
                          alpha_vals, n_e_vals) -> list[str]:
    """Check W1: is the exponent clip active for any reaction?

    Parameters
    ----------
    eta_raw : float
        Raw overpotential (phi_applied - E_eq, nondim).
    exponent_clip : float
    bv_exp_scale : float
    alpha_vals : list[float]
        Per-reaction transfer coefficients.
    n_e_vals : list[int]
        Per-reaction electron counts.

    Returns list of warning strings (empty if no saturation).
    """
    # W1: |bv_exp_scale * eta_raw| >= exponent_clip
    # Also check per-reaction: |alpha * n_e * eta_clipped| near clip
```

---

## File 1: `Forward/bv_solver/solvers.py`

**Functions to modify**: `forsolve_bv`, `solve_bv_with_continuation`, `solve_bv_with_ptc`, `solve_bv_with_charge_continuation`

### Changes:

#### `forsolve_bv()` (return at line 72)
- **Before `return U`**: Call `validate_solution_state(U, ...)` using ctx metadata.
- Extract `n_species`, `c_bulk` (from `c0`), `phi_applied`, `z_vals`, `eps_c` from solver_params/ctx.
- If `valid == False`: log failures, raise `PhysicsViolationError` (or return tagged result).
- If warnings: emit via `warnings.warn()`.
- **Checks**: F1, F4, F6, W2, W3, W5

#### `solve_bv_with_continuation()` (return at line 213)
- Same pattern before `return ctx["U"]`.
- **Checks**: F1, F4, W2, W3

#### `solve_bv_with_ptc()` (return at line 394)
- Same pattern before `return ctx["U"]`.
- **Checks**: F1, F4, W2, W3

#### `solve_bv_with_charge_continuation()` (return at lines 567-568)
- Before return, validate the final state.
- Also check `achieved_z_factor` (F5).
- **Checks**: F1, F4, F5, W2, W3

---

## File 2: `Forward/bv_solver/grid_charge_continuation.py`

**Functions to modify**: `solve_grid_with_charge_continuation` (Phase 1 + Phase 2 loops)

### Changes:

#### Phase 1 neutral snapshot (line 403)
- After `neutral_solutions[orig_idx] = tuple(...)`:
- Call `validate_solution_state(ctx["U"], ...)` with z=0 context.
- On failure: mark point as failed, skip from Phase 2.
- **Checks**: F1, W2 (neutral regime, so F4/F6 less likely)

#### Phase 2 z-ramp result (lines 560-572)
- After `U_data = tuple(...)` and before `GridPointResult(...)` creation:
- Call `validate_solution_state(ctx["U"], ...)` with full z context.
- On failure: set `converged=False` regardless of achieved_z.
- **Checks**: F1, F4, F5, F6, W2, W3, W5

#### GridPointResult dataclass (line 35)
- Add optional field: `validation: ValidationResult | None = None`
- Store validation result for downstream inspection.

#### GridChargeContinuationResult (line 46)
- Add method: `physics_failures() -> list[GridPointResult]` (points that failed validation).

---

## File 3: `Forward/bv_solver/robust_forward.py`

**Functions to modify**: `_z_ramp_worker`, `solve_curve_robust`, `populate_ic_cache_robust`

### Changes:

#### `_z_ramp_worker()` (result dict at lines 456-465)
- After `U_data = tuple(...)` at line 451:
- Validate solution state. On failure, set `converged=False`.
- After observable assembly (cd, pc): call `validate_observables(cd, pc, I_lim=..., ...)`.
- On F2/F3/F7 failure: set `converged=False`.
- **Checks**: F1, F2, F3, F4, F5, F7, W2, W3

#### `solve_curve_robust()` (return at lines 581-587)
- After collecting all worker results, log aggregate validation stats.
- **No new checks here** — validation is per-point in worker.

#### `populate_ic_cache_robust()` (cache population at line 626)
- Before caching: verify that the solution passes validation.
- **Critical**: don't cache invalid solutions, they poison warm-starts.
- **Checks**: F1, F4 (must pass to be cached)

#### RobustCurveResult dataclass (line 52)
- Add field: `validation_failures: np.ndarray` (bool per point, True if physics failed).

---

## File 4: `Forward/bv_solver/hybrid_forward.py`

**Functions to modify**: `_solve_z0_points`, `solve_curve_hybrid`

### Changes:

#### `_solve_z0_points()` (acceptance at line 223)
- Tighten acceptance: require `ok == True`, not just `steps > 0`.
- After observable assembly at line 223: call `validate_observables(cd, pc, ...)`.
- After U snapshot at line 226: call `validate_solution_state(...)`.
- On failure: set cd/pc to NaN, mark unconverged.
- **Checks**: F1, F2, F7, W2

#### `solve_curve_hybrid()` (return at lines 134-140)
- After merging z=0 and z=1 results: log aggregate validation stats.
- **Checks**: aggregate only (per-point checks done above)

---

## File 5: `Forward/bv_solver/observables.py`

**Functions to modify**: `_build_bv_observable_form` (and add a new `assemble_and_validate` helper)

### Changes:

#### New function: `assemble_observable_validated(form, *, I_lim, phi_applied, V_T, mode)`
- Wraps `fd.assemble(form)`.
- Calls `validate_observables()` on the result.
- Returns `(value, validation_result)`.
- **Checks**: F2, F3, F7

This gives every caller a single function for "assemble + validate" instead of
requiring each caller to do its own validation.

---

## File 6: `Forward/bv_solver/forms.py`

**Functions to modify**: `build_forms` (add diagnostic metadata to ctx)

### Changes:

#### After building eta_clipped (line 251)
- Store the raw (unclipped) parameters in ctx so downstream code can check W1:
  ```python
  ctx["_diag_bv_exp_scale"] = float(bv_exp_scale)
  ctx["_diag_exponent_clip"] = float(conv_cfg["exponent_clip"])
  ctx["_diag_eps_c"] = float(conv_cfg["conc_floor"])
  ```

#### After building c_surf (line 297)
- Store `eps_c` in ctx for downstream floor-domination checks.
- Already partially done via `ctx["bv_convergence"]`, but make it explicit.

#### In the multi-reaction loop (lines 328-379)
- Store per-reaction E_eq values in ctx:
  ```python
  ctx["_diag_E_eq_per_reaction"] = [float(rxn["E_eq_v"]) for rxn in rxns_scaled]
  ```
- This enables downstream W1 checking without re-parsing config.

**Note**: forms.py builds UFL expressions (symbolic). Actual value checks happen at
solve time (solvers.py) or assembly time (observables.py). This file only needs to
expose the metadata.

---

## File 7: `FluxCurve/bv_point_solve/__init__.py`

**Functions to modify**: `solve_bv_curve_points_with_warmstart`

### Changes:

#### After observable assembly (line 622, convergence check)
- Call `validate_observables(simulated_flux, ...)`.
- If F2/F7: mark point failed immediately, don't continue time-stepping.

#### After steady-state convergence (line 663)
- Before building PointAdjointResult: call `validate_solution_state(U, ...)`.
- If F1/F4: set `converged=False`, return fail result.
- **Checks**: F1, F2, F4, F7, W8

#### After adjoint assembly (line 677)
- Validate the annotated flux value before computing objective.
- **Checks**: F2, F3

---

## File 8: `FluxCurve/bv_point_solve/forward.py`

**Functions to modify**: `_solve_cached_fast_path`

### Changes:

#### After observable assembly (line 242)
- Call `validate_observables(...)`.
- If failure: return None (fallback to sequential).
- **Checks**: F2, F7

#### Before cache update (line 300)
- Validate solution state before caching.
- **Critical**: bad solutions in cache propagate to all subsequent warm-starts.
- **Checks**: F1, F4

---

## File 9: `FluxCurve/bv_curve_eval.py`

**Functions to modify**: `evaluate_bv_curve_objective_and_gradient`, `evaluate_bv_multi_observable_objective_and_gradient`

### Changes:

#### Per-point aggregation (line 75)
- After extracting `simulated_flux[i]`: check F2, F7.
- On failure: skip point from objective (treat as missing data).

#### Multi-observable evaluation (lines 289-296, 315-322)
- After primary and secondary point extraction:
- Check F3 (selectivity) on the pair.
- **Checks**: F2, F3, F7

---

## File 10: `Surrogate/training.py`

**Functions to modify**: `generate_training_data_single`, `_interpolate_failed_points`

### Changes:

#### After extracting cd_flux and pc_flux (lines 146, 175)
- Call `validate_observables(cd, pc, ...)` per voltage point.
- On failure: mark that point as unconverged in mask.
- **Checks**: F2, F3, F7

#### After interpolation (line 233)
- Re-validate interpolated values.
- **Checks**: F2, F3 (interpolation can't fix wrong-sign or over-limit)

---

## File 11: `scripts/surrogate/overnight_train_v16.py`

**Functions to modify**: `_validate_sample`

### Changes:

#### Extend existing validation (lines 377-398)
Currently checks: z >= 0.999, finite observables. Add:
- F1: Check stored U_data for negative concentrations (requires loading solution state).
- F2: Check cd values against I_lim.
- F3: Check pc/cd selectivity.
- F4: Check surface concentrations at electrode boundary against eps_c.
- F7: Check pc sign at cathodic voltages.
- W1: Check if any voltage point is clip-saturated.
- **Checks**: F1, F2, F3, F4, F7, W1

---

## File 12: `scripts/surrogate/compute_adjoint_gradients_v16.py`

**Functions to modify**: `_load_forward_data`, `_compute_gradients_single_point`

### Changes:

#### `_load_forward_data()` (line 422)
- After loading converged array: re-validate stored solutions.
- Don't trust the training pipeline's convergence flag blindly.
- **Checks**: F1, F4 (on stored U_data)

#### `_compute_gradients_single_point()` (line 257)
- After adjoint derivative: validate gradient magnitudes.
- W: warn if any gradient is NaN, Inf, or suspiciously large (>1e6).

---

## Implementation Order

**Wave 1** (foundation, no dependencies):
- File 0: `validation.py` — shared module

**Wave 2** (core solver, depends on Wave 1):
- File 1: `solvers.py`
- File 2: `grid_charge_continuation.py`
- File 5: `observables.py`
- File 6: `forms.py`

**Wave 3** (higher-level solvers, depends on Wave 2):
- File 3: `robust_forward.py`
- File 4: `hybrid_forward.py`
- File 7: `bv_point_solve/__init__.py`
- File 8: `bv_point_solve/forward.py`

**Wave 4** (pipeline consumers, depends on Wave 3):
- File 9: `bv_curve_eval.py`
- File 10: `Surrogate/training.py`
- File 11: `overnight_train_v16.py`
- File 12: `compute_adjoint_gradients_v16.py`
