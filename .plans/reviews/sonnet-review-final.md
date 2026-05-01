# Plan Review (Final)

**Verdict:** APPROVED
**Reviewer:** Sonnet

---

## API Verification Checklist

### 1. `solve_grid_with_charge_continuation` — signature

**Actual signature** (`grid_charge_continuation.py:81-91`):
```python
def solve_grid_with_charge_continuation(
    solver_params,
    *,
    phi_applied_values: np.ndarray,
    charge_steps: int = 10,
    mesh: Any = None,
    min_delta_z: float = 0.005,
    max_eta_gap: float = 3.0,
    print_interval: int = 20,
    per_point_callback: Optional[Callable] = None,
) -> GridChargeContinuationResult:
```

**Plan uses** (Step 2 target gen):
```python
solve_grid_with_charge_continuation(
    sp,
    phi_applied_values=phi_applied_values,
    charge_steps=charge_steps,
    mesh=mesh,
    max_eta_gap=max_eta_gap,
    per_point_callback=_extract_observables,
)
```

**Plan uses** (Step 3 voltage extension):
```python
solve_grid_with_charge_continuation(
    sp,
    phi_applied_values=all_eta,
    charge_steps=charge_steps,
    mesh=mesh,
    max_eta_gap=max_eta_gap,
)
```

MATCH. All kwargs are valid. `min_delta_z` and `print_interval` are correctly left at defaults.

---

### 2. `per_point_callback` call site and signature

**Actual call site** (`grid_charge_continuation.py:563-564`):
```python
if per_point_callback is not None:
    per_point_callback(orig_idx, eta_i, ctx)
```

**Plan claims** (Step 2, Note): `callback(orig_idx, phi_applied, ctx)` — verified in `grid_charge_continuation.py:564`.

**Plan's callback definition**:
```python
def _extract_observables(orig_idx, phi_applied, ctx):
```

MATCH. Signature is correct.

---

### 3. `GridChargeContinuationResult` attributes

**Actual** (`grid_charge_continuation.py:47-60`):
- `points: Dict[int, GridPointResult]` — MATCH (plan iterates `result.points.items()`)
- `mesh_dof_count: int` — MATCH (plan references `result.mesh_dof_count` in Step 3 comment; not actually used as attribute in the plan's code since `populate_cache_entry` takes it as a third arg)
- `all_converged() -> bool` — MATCH (plan calls `result.all_converged()`)
- `partial_points() -> list` — MATCH (plan calls `result.partial_points()`)

---

### 4. `GridPointResult` attributes

**Actual** (`grid_charge_continuation.py:36-43`):
- `orig_idx: int` — MATCH
- `phi_applied: float` — MATCH
- `U_data: tuple` — MATCH (plan accesses `pt.U_data`)
- `achieved_z_factor: float` — MATCH
- `converged: bool` — MATCH (plan checks `pt.converged`)

---

### 5. `_build_bv_observable_form` — signature

**Actual** (`observables.py:13-19`):
```python
def _build_bv_observable_form(
    ctx: Dict[str, object],
    *,
    mode: str,
    reaction_index: Optional[int],
    scale: float,
) -> object:
```

**Plan uses** (Step 2):
```python
_build_bv_observable_form(
    ctx, mode="current_density", reaction_index=None, scale=observable_scale,
)
_build_bv_observable_form(
    ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale,
)
```

MATCH. `reaction_index=None` is valid for `"current_density"` and `"peroxide_current"` modes (the function only raises if `mode="reaction"` and `reaction_index is None`).

---

### 6. Import path for `_build_bv_observable_form`

**Plan specifies**: `from Forward.bv_solver.observables import _build_bv_observable_form`

**Actual location**: `Forward/bv_solver/observables.py` — CONFIRMED. The plan explicitly notes to use the canonical path rather than `FluxCurve.bv_observables` shim. This is correct — v15 uses the shim (`from FluxCurve.bv_observables import _build_bv_observable_form`), but v16 should use the canonical path. Both resolve to the same function, so either works; the canonical path is cleaner.

---

### 7. `solve_grid_with_charge_continuation` exported from `Forward.bv_solver`

**Actual** (`Forward/bv_solver/__init__.py:70-74`):
```python
from Forward.bv_solver.grid_charge_continuation import (
    solve_grid_with_charge_continuation,
    GridChargeContinuationResult,
    GridPointResult,
)
```

MATCH. `from Forward.bv_solver import solve_grid_with_charge_continuation` works.

---

### 8. `populate_cache_entry` — signature

**Actual** (`FluxCurve/bv_point_solve/cache.py:69`):
```python
def populate_cache_entry(orig_idx: int, U_data: tuple, mesh_dof_count: int) -> None:
```

**Plan uses**:
```python
populate_cache_entry(idx, pt.U_data, result.mesh_dof_count)
```

MATCH. All three positional arguments are provided correctly.

---

### 9. `mark_cache_populated_if_complete` — signature

**Actual** (`FluxCurve/bv_point_solve/cache.py:86`):
```python
def mark_cache_populated_if_complete(n_points: int) -> bool:
```

**Plan uses**:
```python
mark_cache_populated_if_complete(len(all_eta))
```

MATCH.

---

### 10. `populate_cache_entry` / `mark_cache_populated_if_complete` import path

**Plan says**: `from FluxCurve.bv_point_solve import populate_cache_entry, mark_cache_populated_if_complete`

**Actual exports** (`FluxCurve/bv_point_solve/__init__.py:53-54, 89-90`): both names are in `__all__` and explicitly imported from `cache.py`.

MATCH.

---

### 11. `make_bv_solver_params` kwargs used in Step 3

**Actual signature** (`scripts/_bv_common.py:354-370`):
```python
def make_bv_solver_params(*, eta_hat, dt, t_end, species, snes_opts,
                           ..., k0_hat_r1, k0_hat_r2, alpha_r1, alpha_r2, ...)
```

**Plan uses** (Step 3):
```python
sp = make_bv_solver_params(
    eta_hat=0.0, dt=0.5, t_end=50.0,
    species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
    k0_hat_r1=float(warm_k0[0]), k0_hat_r2=float(warm_k0[1]),
    alpha_r1=float(warm_alpha[0]), alpha_r2=float(warm_alpha[1]),
)
```

MATCH. All kwargs are valid named parameters.

---

### 12. `_target_cache_path` — existing signature in v15

**Actual v15 signature** (`Infer_BVMaster_charged_v15.py:156`):
```python
def _target_cache_path(phi_applied_values, observable_scale, eta_steps, charge_steps):
```

**Plan proposes changing to**:
```python
def _target_cache_path(phi_applied_values, observable_scale, max_eta_gap, charge_steps):
```

And the hash line changes from:
```python
f"eta_steps={eta_steps},charge_steps={charge_steps}".encode()
```
to:
```python
f"method=unified_grid,max_eta_gap={max_eta_gap},charge_steps={charge_steps}".encode()
```

Correct approach. The new hash ensures no cache collision with v15's entries.

---

### 13. `_generate_targets_with_charge_cont` — call site threading

**Actual v15 caller** (`Infer_BVMaster_charged_v15.py:612-615`):
```python
targets = _generate_targets_with_charge_cont(
    all_eta, observable_scale, args.noise_percent, args.noise_seed,
    args.eta_steps, args.charge_steps,
)
```

**Plan says**: Update caller to pass `args.max_eta_gap` instead of `args.eta_steps`.

Confirmed: v15 has `args.eta_steps` at this call site. The plan correctly identifies it and replaces it with `args.max_eta_gap`. The plan also says to keep `--eta-steps` as a deprecated no-op (Step 5), so old invocations won't hard-error.

---

### 14. `_extend_voltage_cache_for_p2` — `base_sp` parameter

**Actual v15 signature** (`Infer_BVMaster_charged_v15.py:386-394`):
```python
def _extend_voltage_cache_for_p2(
    all_eta, warm_k0, warm_alpha, base_sp,
    *, eta_steps=20, charge_steps=10,
) -> int:
```

**Plan's v16 signature** (Step 3):
```python
def _extend_voltage_cache_for_p2_unified(
    all_eta, warm_k0, warm_alpha, *, charge_steps=10, max_eta_gap=3.0,
):
```

The `base_sp` parameter is dropped from the new signature. The plan notes (in the previous round fix) that `base_sp` was unused in the hot path — v15 rebuilds `sp` from scratch inside the loop using `make_bv_solver_params`. Verified: the actual v15 code does not use `base_sp` anywhere in the function body (lines 386-475 were read above; `base_sp` appears in the signature but is never referenced in the body). Dropping it is correct.

---

### 15. `eta_hat=0.0` in both Step 2 and Step 3 `make_bv_solver_params` calls

**Plan**: Both Step 2 (target gen) and Step 3 (voltage extension) call `make_bv_solver_params(eta_hat=0.0, ...)`. The voltage sweep is driven by `phi_applied_values` passed to `solve_grid_with_charge_continuation`, which overrides the `phi_applied` field internally (confirmed at `grid_charge_continuation.py:142-145`). Setting `eta_hat=0.0` as a placeholder is correct.

MATCH.

---

### 16. `t_end=50.0` vs. `dt * max_ss_steps`

**Plan**: Both steps hard-code `dt=0.5, t_end=50.0`.

**v15**: Computes `t_end = dt * max_ss_steps` where `max_ss_steps=100`, giving `t_end = 50.0`.

Numerically identical. No issue.

---

## Summary

All 16 API checkpoints pass against the actual codebase. Every function signature, kwarg name, return type attribute, import path, and call-site argument is verified correct. The plan accurately describes the delta from v15, correctly identifies what changes are needed in `_target_cache_path`, `_generate_targets_with_charge_cont`, `_extend_voltage_cache_for_p2`, and the CLI argument parser. The `base_sp` drop from the extension function is safe (the parameter existed but was dead code in v15). The `adj.stop_annotating()` guard is preserved in both new functions. The cache invalidation strategy (new hash key) correctly prevents stale cache reuse across v15/v16 runs.

---

## Major Issues

None found.

---

## Minor Issues

None found.

---

## Implementation Readiness

Yes. This plan is ready for direct implementation. An implementer can copy v15 to v16 and apply Steps 2–6 exactly as written without needing to fill any gaps, resolve any ambiguities, or look up any additional API details. All function signatures, import paths, attribute names, and call-site arguments have been independently verified against the live codebase.
