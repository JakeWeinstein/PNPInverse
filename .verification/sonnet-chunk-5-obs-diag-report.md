# Verification Report: Observables, Diagnostics, Validation, Public API
**Scope:** `Forward/bv_solver/validation.py`, `diagnostics.py`, `observables.py`, `__init__.py`
**Date:** 2026-05-05
**Reviewer:** Claude Sonnet 4.6 (independent subagent)

---

## Summary

The four files are largely correct for the production use case. Three issues reach **warning** severity; none reach **critical**. The most significant finding is that `exponent_clip` is a declared parameter of `validate_solution_state` that is silently unused — neither W1 (clip saturation check) nor any other check consumes it. Several other gaps are documented below.

---

## Issues Found

---

### ISSUE 1

**SEVERITY:** warning

**LOCATION:** `Forward/bv_solver/validation.py:53` — `exponent_clip: float` parameter

**DESCRIPTION:**
`exponent_clip` is declared as a required keyword-only argument to `validate_solution_state` but is never referenced in the function body. The module docstring lists W1 ("Clip saturation — BV exponent near clipping threshold") as one of the implemented warning codes, and the docstring example shows `"W1: clip saturated at V=0.3"`. Neither the W1 check nor any other check that reads `exponent_clip` exists anywhere in the 202-line function. The parameter is accepted and silently discarded.

**EVIDENCE:**
```
grep -n "exponent_clip" Forward/bv_solver/validation.py
→ 53:    exponent_clip: float,    # only occurrence — never read below this line
```
All callers (`FluxCurve/bv_point_solve/__init__.py:722`, `bv_point_solve/forward.py:317`, `scripts/studies/v18_test_3species_boltzmann.py:213`) faithfully pass the value (sourced from `ctx["_diag_exponent_clip"]`), so the public interface is stable. But the clip-saturation warning that the parameter was intended to enable is dead code. Per CLAUDE.md, the clip threshold at `exponent_clip=100` (current production default) is "the only PC-trustworthy setting"; an active W1 would warn when the solver approaches the threshold. Without it, callers get no warning about clip-limited solutions from `validate_solution_state`.

**NOTE:** W8 ("Bulk integral still drifting despite flux steady-state") is also listed in the docstring classification table but never implemented. These two missing checks are the only significant gap in `validation.py`.

---

### ISSUE 2

**SEVERITY:** warning

**LOCATION:** `Forward/bv_solver/validation.py:181–195` — W5 (cation depletion) — DOF size mismatch risk

**DESCRIPTION:**
W5 indexes concentration data with a boolean mask derived from mesh coordinate DOFs:

```python
coords = U.function_space().mesh().coordinates.dat.data_ro   # shape: (n_nodes,)
y_coords = coords[:, -1] if coords.ndim == 2 else coords      # last column or identity
top_mask = y_coords >= y_median                               # shape: (n_nodes,)
c_data = _conc_array(i)                                       # shape: (n_dofs_per_species,)
c_top_min = float(c_data[top_mask].min())                     # applied across c_data
```

For CG1 elements on a 1D `IntervalMesh` or a 2D `RectangleMesh`, the vertex count equals the DOF count per species, so the mask sizes agree. However, for CG2 or higher-order elements (`order=2` in `build_context_logc`), the DOF count per species exceeds the vertex count and the mask shapes diverge. In that case `c_data[top_mask]` silently applies a too-short boolean index, raising either a `ValueError` at runtime or producing a subtly wrong result depending on numpy broadcasting.

**EVIDENCE:**
`forms_logc.py:64`: `V_scalar = fd.FunctionSpace(mesh, "CG", order)` where `order` is passed from `solver_params`. The default order appears to be 1 in practice, but the code does not guard against order > 1. Firedrake's coordinate function space is always CG1/P1 regardless of the solution space order.

**NOTE:** For the current production stack (order=1, 1D or 2D mesh), the shapes agree and W5 is safe. This is a latent bug only triggered by order > 1.

---

### ISSUE 3

**SEVERITY:** warning

**LOCATION:** `Forward/bv_solver/validation.py:318–319` — `is_logc` flag not set for muh contexts in production callers

**DESCRIPTION:**
The two production callers that reach `validate_solution_state` (`FluxCurve/bv_point_solve/__init__.py:723` and `FluxCurve/bv_point_solve/forward.py:318`) pass:

```python
is_logc=bool(ctx.get("logc_transform", False)),
```

This is correct for the `logc` backend (`logc_transform=True` is stored in ctx by `forms_logc.py:536`). The `logc_muh` backend also sets `logc_transform=True` (forms_logc_muh.py:579), so `is_logc` will be `True` for muh contexts as well. **However, neither caller passes `mu_species=ctx.get("mu_species")` or `em=ctx["nondim"].get("electromigration_prefactor", 1.0)`** — the two muh-specific keyword arguments documented in CLAUDE.md and the function docstring.

Consequence: for a muh-formulation solution, the validator reads raw `U.dat[mu_h_idx]` (which is `mu_H = u_H + em*z_H*phi`, not `u_H`) and applies `exp(raw)` without the phi correction. The recovered concentration for H+ is `exp(mu_H)` instead of `exp(mu_H - em*z_H*phi) = c_H`. Since `em*z_H*phi` can be O(10) in nondim units across the Debye layer, the reported H+ concentrations used in F1 and W5 checks are up to `exp(20)` = 5×10⁸ times the physical value, making the checks meaningless for that species.

**EVIDENCE:**
- `forms_logc_muh.py:578`: `ctx["mu_species"] = list(mu_species)` is stored
- `CLAUDE.md:166-168`: explicitly states `mu_species=ctx.get('mu_species')` and `em=...` must be passed
- Neither `bv_point_solve/__init__.py` nor `bv_point_solve/forward.py` passes these; `grid_per_voltage.py` does not call `validate_solution_state` at all

The study scripts (`v18_test_3species_boltzmann.py`, `v18_test_3sp_logc.py`) also omit these args but they use the `logc` backend, not `logc_muh`, so the omission is benign there.

---

### ISSUE 4 (NOTE)

**SEVERITY:** note

**LOCATION:** `Forward/bv_solver/validation.py:107–131` — F1 and F4 in logc mode

**DESCRIPTION:**
The docstring accurately notes that F1 (negative concentration) is "structurally unreachable" in logc mode since `exp(u) > 0`. The loop body is retained "for symmetry." This is not a bug but is worth confirming: the check `c_min < -eps_c` will never trigger because `exp(anything) > 0 > -eps_c`. The loop is dead code in logc mode and adds negligible cost. F4 is correctly gated by `if not is_logc:`.

---

## Diagnostics (`diagnostics.py`)

### Finding D1 (Correct)

**Surface field means** (`surface_field_means`): The function correctly handles both the logc and muh formulations. For muh contexts it reads `u_exprs[i]` (the reconstructed `log(c_i) = mu_H - em*z_H*phi`) from `ctx["u_exprs"]`, which is populated by `forms_logc_muh.py:577`. This gives semantically consistent `u{i}_surface_mean` (= surface mean of `log(c_i)`) across both backends. The additional `mu{i}_surface_mean` (raw muh primary variable mean) is correctly reported for the muh species.

The `c{i}_surface_mean = exp(u_mean)` Jensen-fault (exp of mean ≠ mean of exp) is documented inline ("retains the existing `exp(mean(u))` Jensen-fault ... a separate correctness fix is tracked"). This is a known approximation, not a bug.

### Finding D2 (Correct)

**Bikerman steric saturation** (`check_steric_saturation`, `collect_diagnostics`): The Bikerman closure computed in diagnostics is:

```
theta_b = 1 - A_dyn_bulk - a_b * c_bulk_b
c_surf  = c_bulk * exp(-z*phi) * (1 - A_dyn_surf) / (theta_b + a_b*c_bulk*exp(-z*phi))
```

This matches the formula in `docs/steric_analytic_clo4_reduction_handoff.md`. The `phi_clamp` guard prevents `exp(-z*phi)` overflow. The fallback `A_dyn_surf = A_dyn_bulk` on missing surface data is conservative (overestimates packing, underestimates c_surf, meaning steric violations may be missed for high-anodic conditions). The `within_steric` flag is set correctly; violations emit a `UserWarning`.

**SEVERITY:** note — The `A_dyn_surf = A_dyn_bulk` fallback means steric violations at high anodic phi can be silently missed when `c{i}_surface_mean` overflows to `inf` (the `not np.isfinite(c_i_surf)` branch). This is a best-effort diagnostic on the failure path.

### Finding D3 (Note — no Stern consistency or mass balance check)

**DESCRIPTION:**
The task specification requires: "Stern consistency: surface charge sigma = C_S · (phi_metal − phi_solution)" and "Mass balance: ∫_Ω r_i dx − ∫_∂Ω_electrode J_i·n dA = 0." Neither check exists in `diagnostics.py`. The module computes surface field means, SNES reason/iteration counts, and Bikerman steric saturation — but no Faradaic current density at the electrode, no mass balance residual, and no Stern surface charge check.

This is a **scope gap** relative to the intent described in the task, not a correctness error in what is implemented. The file docstring accurately states it is for "failure-mode information (max phi, surface concentrations, SNES reason / iters, Bikerman steric saturation)." The mass balance and Stern consistency checks were apparently planned (they appear in `docs/physics_validation_plan.md`) but not yet implemented.

**SEVERITY:** note — Absent functionality, not incorrect logic.

---

## Observables (`observables.py`)

### Finding O1 (Correct)

**Surface integration on correct boundary:** `_build_bv_observable_form` builds `fd.Measure("ds", domain=ctx["mesh"])` and integrates over `electrode_marker` from `ctx.get("bv_settings", {}).get("electrode_marker", 1)`. Both `forms_logc.py` and `forms_logc_muh.py` store `bv_cfg` into `ctx["bv_settings"]` before returning. The BV residual in those modules also integrates over the same `electrode_marker`. The `bv_rate_exprs` are UFL expressions defined with the same mesh, so no marker mismatch is possible.

The default fallback `electrode_marker=1` matches the interval mesh convention (marker 1 = left = electrode). For the rectangle mesh the electrode is marker 3 (bottom), which is explicitly set in `_bv_common.make_bv_bc_config` (electrode_marker=3, default). Since callers must build a ctx through `build_forms_logc`, which stores `bv_cfg["electrode_marker"]` into `ctx["bv_settings"]`, the fallback is never reached in practice — the correct marker is always present.

### Finding O2 (Correct — with note)

**Peroxide current formula:** The observable uses `scale * (R_0 - R_1) * ds`, where `R_0` is the R1 reaction rate (O₂ → H₂O₂, cathodic) and `R_1` is the R2 reaction rate (H₂O₂ → H₂O, further reduction). This formula gives the net flux of H₂O₂ out of the electrode: R1 produces it (R_0 > 0 cathodic), R2 consumes it (R_1 > 0 cathodic). So `R_0 - R_1 > 0` when more H₂O₂ is produced than consumed, which with `scale = -I_SCALE` (used by all production scripts) gives a negative number for a net cathodic H₂O₂ flux. This is consistent with the sign convention used throughout (`cd < 0` for cathodic current).

**NOTE:** The formula does not weight by `n_electrons` per reaction. The `I_SCALE = n_e * F * D_ref * c_scale / L_ref * 0.1` already folds in `n_e=2` as a global factor, and both reactions have `n_electrons=2` in the production config. The rate expressions `R_j` themselves are dimensionless flux densities in velocity units (nondim); the `n_e` factor in `I_SCALE` converts them to current density. This is self-consistent. If reactions with different `n_electrons` values were introduced (not the current production config), the global `I_SCALE` factor would produce an incorrect peroxide current because the formula treats both reactions as having the same electron stoichiometry. This is a latent design issue, not a current correctness bug.

### Finding O3 (Correct)

**`assemble_observable_validated`:** The function correctly performs only F2 (magnitude vs. I_lim) for single-observable assembly, explicitly documenting that F3 and F7 require both `cd` and `pc` and happen at the pipeline level. The comment in `docs/physics_validation_log.md` item 6 confirms a previous design where `pc=0` was incorrectly passed for the F7 check was corrected to this honest single-observable F2. The current implementation is correct.

### Finding O4 (Note — `bv_curve_eval` passes `pc=0.0` to `validate_observables`)

**LOCATION:** `FluxCurve/bv_curve_eval.py:80-83`

```python
vr = validate_observables(
    float(point.simulated_flux), 0.0,   # ← pc=0.0 always
    I_lim=_i_lim, phi_applied=..., V_T=1.0,
)
```

When `bv_curve_eval` runs in single-observable mode (primary curve only, not multi-reaction), it calls `validate_observables` with `pc=0.0`. With `pc=0.0`:
- F3 (`abs(0/cd) = 0 < 1.05`): always passes — correct, no selectivity check possible
- F7 (`0 > abs(cd)*0.01`?): false when `cd < 0` — always passes — correct for 0

So the behavior is correct: F2 fires on magnitude, F3/F7 are vacuously satisfied. The real selectivity check happens at lines 296-309 in the multi-reaction path where both primary and secondary fluxes are available. This is a design choice made deliberately.

---

## Public API (`__init__.py`)

### Finding A1 (Correct)

**All re-exported names exist.** Every name in `__all__` (lines 94–114) is imported from a module that defines it:

| Symbol | Source module | Verified |
|--------|--------------|---------|
| `make_graded_interval_mesh`, `make_graded_rectangle_mesh` | `mesh.py` | Yes |
| `build_context`, `build_forms`, `set_initial_conditions` | `dispatch.py` | Yes — wrappers over logc/muh |
| `build_context_logc`, `build_forms_logc`, `set_initial_conditions_logc`, `set_initial_conditions_debye_boltzmann_logc` | `dispatch.py` → `forms_logc.py` | Yes |
| `build_context_logc_muh`, `build_forms_logc_muh`, `set_initial_conditions_logc_muh`, `set_initial_conditions_debye_boltzmann_logc_muh` | `dispatch.py` → `forms_logc_muh.py` | Yes — all four defs present |
| `add_boltzmann_counterion_residual`, `build_steric_boltzmann_expressions`, `StericBoltzmannBundle` | `boltzmann.py` | Yes (imported at line 68–72) |
| `solve_grid_per_voltage_cold_with_warm_fallback`, `PerVoltageContinuationResult`, `PerVoltagePointResult` | `grid_per_voltage.py` | Yes |

### Finding A2 (Note — private helpers imported but not in `__all__`)

**LOCATION:** `__init__.py:58–67` — four `_get_bv_*` functions and two `_add_bv_*` functions are imported from `config.py` and `nondim.py` at module level but are not listed in `__all__`.

```python
from Forward.bv_solver.config import (
    _get_bv_cfg,                        # private
    _get_bv_convergence_cfg,            # private
    _get_bv_reactions_cfg,              # private
    _get_bv_boltzmann_counterions_cfg,  # private
)
from Forward.bv_solver.nondim import (
    _add_bv_scaling_to_transform,       # private
    _add_bv_reactions_scaling_to_transform,  # private
)
```

These are not in `__all__`, so `from Forward.bv_solver import *` will not expose them. However, they are accessible as `Forward.bv_solver._get_bv_cfg`, etc. Importing private helpers at the package `__init__` level is an anti-pattern (creates a second reference that prevents `config.py` module cleanup and pollutes tab-completion). It appears these imports were needed by an earlier version of `__init__.py` that also contained config logic; the code was split out but the imports were left behind.

**SEVERITY:** note — Not a correctness error; no broken exports.

---

## Key Correctness Arguments (what is working correctly)

1. **muh reconstruction in `validate_solution_state`:** When called with `is_logc=True`, `mu_species=[h_idx]`, and the correct `em`, the `_conc_array` closure correctly recovers `c_H = exp(mu_H - em*z_H*phi_dof)` elementwise from DOF data. The indexing `U.dat[n_species]` for phi is correct (phi is the last subfuntion in the mixed space, stored at index `n_species`). The logic `if i in mu_set` is correct.

2. **NaN/Inf detection via numpy:** The validator operates on `.dat[i].data_ro` arrays (numpy). If a Newton solve produces NaN DoFs, `float(arr.min())` returns NaN, and `NaN < -eps_c` evaluates to False in Python — so F1 would NOT catch NaN. However, F6 and W2 compare with a NaN maximum: `NaN > threshold` → False, so those also silently pass. The docstring says "no FEM assembly" and uses numpy on DOF data, which is appropriate for CG1. But **NaN DOFs are not detected by any check in `validate_solution_state`.** This is a secondary gap (the SNES convergence check upstream already detects NaN in the residual, but a check in the validator would add defense-in-depth). This is a note, not a critical issue — in practice NaN would cause SNES divergence before validation is reached.

3. **Mass balance for Boltzmann counterion:** The Bikerman/ideal analytic counterion is never passed to `_conc_array` because it is not a DOF in `U` — it is an analytic expression in the Poisson residual. Diagnostics correctly handles this by evaluating the counterion surface concentration analytically from `phi_surface_mean` rather than trying to access DOF data. There is no attempt to compute a mass balance for the counterion species (which would require the analytic NP flux, not a DOF array). This is correct by design.

4. **`__init__.py` dispatch chain:** `build_context` / `build_forms` / `set_initial_conditions` correctly delegate to `logc_muh` or `logc` backends via `_resolve_backend(solver_params)`, which reads `params[10]["bv_convergence"]["formulation"]`. Unknown formulations fall through to `logc` (defensive default). The `config.py` layer validates the formulation at parse time. The chain has no broken links.

---

## Summary Table

| # | Severity | File | Description |
|---|----------|------|-------------|
| 1 | warning | `validation.py:53` | `exponent_clip` parameter accepted but never used; W1 check not implemented |
| 2 | warning | `validation.py:181-195` | W5 DOF-mask size mismatch latent bug for order > 1 elements |
| 3 | warning | `bv_point_solve/forward.py:318`, `__init__.py:723` | `mu_species` and `em` not passed for muh contexts; H+ concentration check wrong under muh |
| D3 | note | `diagnostics.py` | Mass balance and Stern consistency checks not implemented (planned, not present) |
| O4 | note | `bv_curve_eval.py:81` | `pc=0.0` in single-mode `validate_observables` call — correct behavior, intentional design |
| A2 | note | `__init__.py:58-67` | Private `_get_bv_*` and `_add_bv_*` helpers imported but not in `__all__`; minor leakage |

The **most urgent fix** is Issue 3: the two production callers of `validate_solution_state` do not pass `mu_species` / `em` for muh contexts, making the H+ concentration checks physically wrong when `formulation="logc_muh"` is active.
