# Second-Pass Verification Report — Chunk 2
**Verifier:** claude-sonnet-4-6 (independent pass)
**Date:** 2026-05-02
**Scope:**
  - `Forward/bv_solver/__init__.py` (149 lines)
  - `Forward/bv_solver/dispatch.py` (125 lines)
  - `Forward/bv_solver/forms_logc.py` (531 lines)

**Context-only (read but not in scope):**
  - `Forward/bv_solver/forms.py`
  - `Forward/bv_solver/grid_per_voltage.py` (patched orchestrator)
  - `Forward/bv_solver/boltzmann.py`
  - `Forward/bv_solver/observables.py`
  - `Forward/bv_solver/nondim.py`
  - `Forward/bv_solver/config.py`
  - `Forward/params.py`

---

## Summary

No new **critical** issues found. The previously reported findings (C6a, C12a, C6b, A4-Q) are re-confirmed as still present and accurately described. All items in the "Additional Things to Look For" checklist from the brief are verified clean, with one new observation and one clarification/upgrade of a prior finding.

---

## Re-verification of Prior Findings

### C6a (CONFIRMED — minor)
**Location:** `forms_logc.py:349–357`
**Description:** In the `bv_log_rate` branch, the `else: anodic = fd.Constant(0.0)` arm covers two cases simultaneously — a truly irreversible reaction (`rxn["reversible"] == False`) AND a reversible reaction where `rxn["c_ref_model"] <= 1e-30`. Both collapse to zero anodic rate, which is correct for the irreversible case but silently ignores the reversible+small-c_ref case. A reviewer cannot tell from the code whether the latter ever occurs in practice. Finding confirmed unchanged.

**Smallest Fix:** Add an explicit guard or a `warnings.warn` for the reversible+near-zero-c_ref path:
```python
else:
    if rxn["reversible"]:
        import warnings
        warnings.warn(
            f"Reaction {j}: reversible but c_ref_model={rxn['c_ref_model']:.2e} <= 1e-30; "
            "treating as irreversible (anodic=0).", stacklevel=3
        )
    anodic = fd.Constant(0.0)
```

---

### C12a (CONFIRMED — minor)
**Location:** `forms_logc.py:448–473`
**Description:** `ctx.update(...)` does not include several diagnostic keys that `forms.py` exposes:
- `forms.py` stores `"bv_scaling"` (the full BV sub-dict of the scaling transform); `forms_logc.py` stores `"nondim"` only (the full scaling dict), which is a superset, but the *key name* differs. Any code referencing `ctx["bv_scaling"]` against a logc context will raise `KeyError`.
- `forms.py` also stores `"bv_stoichiometry"` under the legacy path, which `forms_logc.py` does not expose.

Finding confirmed. The comment in `forms_logc.py` at line 468 ("Diagnostic metadata mirroring forms.py") is partially satisfied but not fully: `ctx["bv_scaling"]` is still absent.

**Smallest Fix:** Add `"bv_scaling": scaling` to the `ctx.update(...)` dict at line 448 for full parity with `forms.py`.

---

### C6b (CONFIRMED — question, benign)
**Location:** `forms_logc.py:364–367`
**Description:** In the non-log-rate path, `power = factor["power"]` is the raw value from the config dict, which `config.py:_get_bv_reactions_cfg` stores as `int(f_cfg.get("power", 1))` (line 247 of config.py). So `power` is always a Python `int`. The expression `(c_surf[sp_idx] / c_ref_f) ** power` therefore uses Python integer exponentiation of a UFL expression, which Firedrake accepts and expands symbolically. In the log-rate path (line 334), `power = fd.Constant(float(factor["power"]))` is used instead. The difference is benign for UFL compilation but inconsistent. If a non-integer power is ever needed, only the log-rate path handles it cleanly.

Finding confirmed and assessed benign for the current integer-only config schema.

---

### A4-Q (CONFIRMED — question)
**Location:** `dispatch.py:55–57`
**Description:** When `solver_params` is a `SolverParams` dataclass, `_params_dict` reads `solver_params.solver_options`. `SolverParams.__post_init__` does not normalize `solver_options` to a dict (it uses `field(default_factory=dict)` so the default is fine, but a caller passing `solver_options=None` explicitly would bypass `default_factory`). In that case `opts = solver_params.solver_options` is `None`, which is not a dict, so `_params_dict` returns `{}`. This then causes `_get_formulation` to return `"concentration"` silently, even if the user intended `"logc"`.

Confirmed still present. In practice, `SolverParams` construction always provides a dict (the `default_factory=dict` covers the no-arg case), so the failure path requires an intentional `solver_options=None`, which is unlikely.

**Smallest Fix (defensive):** In `SolverParams.__post_init__`, add:
```python
object.__setattr__(self, "solver_options", dict(self.solver_options) if self.solver_options is not None else {})
```

---

## New Findings (Additional Items Checklist)

### CHECK 1 — ctx["F_res"] / ctx["J_form"] update order after `add_boltzmann_counterion_residual`

**Result: CLEAN.**

`forms_logc.py:446` computes `J_form = fd.derivative(F_res, U)` from the pre-Boltzmann `F_res`. This J_form is written into ctx at line 450. Then at line 475, `add_boltzmann_counterion_residual(ctx, params)` is called, which (per `boltzmann.py:125–126`) re-derives `J_form` from the updated `F_res` and overwrites both `ctx["F_res"]` and `ctx["J_form"]`. When Boltzmann counterions are configured, the final `ctx["J_form"]` correctly includes the counterion contribution. When no counterions are configured, `add_boltzmann_counterion_residual` returns 0 without touching ctx, so the pre-Boltzmann J_form at line 450 is correct.

The orchestrator (`_build_for_voltage` in `grid_per_voltage.py`) constructs `NonlinearVariationalProblem` after `build_forms` returns, so it always sees the post-Boltzmann `ctx["F_res"]` and `ctx["J_form"]`. **Order is correct.**

---

### CHECK 2 — ctx["bcs"] is a list

**Result: CLEAN.**

`forms_logc.py:441` (Stern path): `bcs = bc_ui + [bc_phi_ground]` — Python list concatenation, result is a list.
`forms_logc.py:444` (non-Stern path): `bcs = bc_ui + [bc_phi_electrode, bc_phi_ground]` — same, result is a list.
`bc_ui` is built as a list via `bc_ui.append(...)` at lines 435–439.
`ctx["bcs"]` is always a Python list. **Correct.**

---

### CHECK 3 — ctx["U"] and ctx["U_prev"] identity

**Result: CLEAN.**

`build_context_logc` creates `U` and `U_prev` as `fd.Function(W)` and stores them in ctx (lines 58–65). `build_forms_logc` retrieves them from ctx at lines 174–175 via `ctx["U"]` and `ctx["U_prev"]`. All UFL `fd.split(U)` and `fd.TestFunctions(W)` calls, all BCs (which reference `W.sub(i)` and `phi_applied_func`), and the Dirichlet BCs all remain bound to the same `U` object. `set_initial_conditions_logc` also retrieves `U_prev = ctx["U_prev"]` and then calls `ctx["U"].assign(U_prev)` at line 512, preserving identity. **Correct.**

---

### CHECK 4 — ctx["phi_applied_func"] is the same mutable R-space Function the orchestrator reassigns

**Result: CLEAN.**

`forms_logc.py:204` creates `phi_applied_func = fd.Function(R_space, name="phi_applied")` and stores it in ctx at line 456. The Dirichlet BC at line 443 (`fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker)`) holds a reference to the same Function object, as does the eta computation at line 227/229. The orchestrator's `_solve_cold` (line 314) calls `ctx["phi_applied_func"].assign(float(V_target_eta))` and `_solve_warm`'s `_march` (line 364) calls `paf.assign(float(v_sub))` — both mutate the same object in place, which is correctly reflected in both the BC and the eta expression without rebuilding any forms. **Correct.**

One nuance confirmed clean: `set_initial_conditions_logc` at line 511 uses `fd.Constant(float(phi_applied_model))` (not `phi_applied_func`) for the initial linear potential profile in `U_prev`. This is intentional and harmless — the IC is a scalar that doesn't need to be a mutable Function.

---

### CHECK 5 — ctx["z_consts"] is a list of mutable fd.Constant objects

**Result: CLEAN.**

`forms_logc.py:172`: `z = [fd.Constant(float(z_vals[i])) for i in range(n)]`

`fd.Constant.assign()` is confirmed working (empirically verified against the venv-firedrake install above). The orchestrator's `_set_z_factor` calls `ctx["z_consts"][i].assign(z_nominal[i] * z_val)` for each species. Since `z` is used in the NP residual (line 276: `drift = em * z[i] * phi`) and Poisson source (line 419: `sum(z[i] * ci[i] * w for i in range(n))`), mutating via `.assign()` correctly updates the live UFL DAG at each z-ramp step. **Correct.**

---

### CHECK 6 — ctx["boltzmann_z_scale"] is a single mutable R-space Function

**Result: CLEAN.**

`boltzmann.py:102–103`: `z_scale = fd.Function(R_space, name="boltzmann_z_scale")` followed by `z_scale.assign(1.0)`. Stored at `boltzmann.py:128`: `ctx["boltzmann_z_scale"] = z_scale`. The orchestrator's `_set_z_factor` reads it via `ctx.get("boltzmann_z_scale")` and calls `.assign(float(z_val))`. `fd.Function` on an R-space (Real element) supports `.assign()` (empirically verified). **Correct.**

When no Boltzmann counterions are configured, `ctx.get("boltzmann_z_scale")` returns `None` and the orchestrator skips the assign — also correct.

---

### CHECK 7 — ctx["dt_const"] is a mutable fd.Constant

**Result: CLEAN.**

`forms_logc.py:201`: `dt_const = fd.Constant(float(scaling["dt_model"]))`. Stored at line 455. The orchestrator's `run_ss` calls `dt_const.assign(dt_val)` at each SS step. `fd.Constant.assign()` confirmed working. The time-derivative residual at line 285 (`((c_i - c_old) / dt_const) * v * dx`) uses this constant as a denominator in the live UFL expression. **Correct.**

---

### CHECK 8 — ctx["bv_rate_exprs"] has exactly two entries for two reactions; order matches observables.py

**Result: CLEAN with a NOTE.**

In `forms_logc.py:305–384`, `bv_rate_exprs` is built by iterating over `rxns_scaled = scaling["bv_reactions"]` (line 306), appending one `R_j = cathodic - anodic` per reaction in the order they appear in `params["bv_bc"]["reactions"]`. The config in the production script configures R1 (O2→H2O2) first and R2 (H2O2→H2O) second, so `bv_rate_exprs[0] = R1`, `bv_rate_exprs[1] = R2`.

`observables.py:52` uses `bv_rate_exprs[0] - bv_rate_exprs[1]` for the `peroxide_current` mode, interpreting this as "production minus consumption." This matches the reaction ordering: R1 produces H2O2 (cathodic current positive for O2 reduction), R2 consumes it. **Order is consistent with the observables contract — provided the caller always configures R1 before R2.**

NOTE: There is no runtime validation that `bv_rate_exprs` has exactly 2 entries before `peroxide_current` mode is used. `observables.py:47–49` does check `len(bv_rate_exprs) < 2` and raises a `ValueError`, so this is guarded. **No issue; the guard exists.**

---

### NEW FINDING N1 — E_eq_j_val == 0.0 check may silently use global E_eq for a reaction with per-reaction E_eq_v = 0.0
**SEVERITY: minor**
**Location:** `forms_logc.py:318–323`

```python
E_eq_j_val = rxn.get("E_eq_model", None)
if E_eq_j_val is not None and E_eq_j_val != 0.0:
    E_eq_j = fd.Constant(float(E_eq_j_val))
    eta_j = _build_eta_clipped(E_eq_j)
else:
    eta_j = eta_clipped
```

`nondim.py` always sets `srxn["E_eq_model"]` (never leaves it absent), so `rxn.get("E_eq_model", None)` will always return a float. The condition `E_eq_j_val != 0.0` means: if a reaction legitimately has `E_eq_v = 0.0` (e.g., a reaction at the standard hydrogen electrode), the code falls through to `eta_clipped`, which uses the **global** `E_eq_model_global = fd.Constant(float(scaling["bv_E_eq_model"]))`.

For the standard production config (R1: E_eq=0.68 V, R2: E_eq=1.78 V), both are non-zero, so this branch never fires. But if a user configures a reaction at E_eq=0.0 V (SHE reference), the per-reaction E_eq is silently ignored and the global one is used instead — which is also 0.0 in that case only if `bv_bc.E_eq_v = 0.0`, but if the two differ (global vs. per-reaction), the wrong eta is used without any warning.

The intent of the `E_eq_j_val != 0.0` check appears to be "fall back to global when no per-reaction E_eq was set." But since nondim always sets `E_eq_model`, the check should instead distinguish "was a per-reaction E_eq explicitly configured." The cleaner sentinel would be `None` (absent from config), not `0.0`.

**Smallest Fix:** Change the condition to check for the source key in the raw config, or use a sentinel value. One clean option — since `nondim._add_bv_reactions_scaling_to_transform` sets `srxn["E_eq_model"] = rxn.get("E_eq_v", 0.0) / potential_scale`, the zero check fails when `E_eq_v` was explicitly set to 0. Fix: track whether the reaction had an explicit per-reaction `E_eq_v` in config.py by storing a boolean flag:

In `config.py:_get_bv_reactions_cfg`, add `"has_per_rxn_E_eq": float(rxn.get("E_eq_v", 0.0)) != 0.0` to each reaction dict (or use a separate sentinel key). Then in `forms_logc.py:318`:

```python
if rxn.get("has_per_rxn_E_eq", False):
    E_eq_j = fd.Constant(float(rxn["E_eq_model"]))
    eta_j = _build_eta_clipped(E_eq_j)
else:
    eta_j = eta_clipped
```

This cleanly handles E_eq_v=0.0 as a genuine zero-overpotential reference.

---

### NEW FINDING N2 — `set_initial_conditions_logc` ignores nondim scaling when building initial phi profile
**SEVERITY: minor**
**Location:** `forms_logc.py:511`

```python
U_prev.sub(n).interpolate(fd.Constant(float(phi_applied_model)) * (1.0 - spatial_var))
```

`phi_applied_model` is the correctly nondimensionalized phi value from `scaling["phi_applied_model"]`. The profile `phi_applied_model * (1 - x)` is a linear ramp from `phi_applied_model` at x=0 (electrode) to 0 at x=1 (bulk). For an `IntervalMesh(N, 0, L)` this is appropriate if the electrode is at x=0 and bulk at x=L≈1.

However, the `build_context_logc` default mesh is `fd.UnitSquareMesh(32, 32)` (line 55), where the spatial coordinate `coords[1]` (y) runs 0..1. The Dirichlet BCs are applied to `electrode_marker` and `concentration_marker` (ground) on different boundaries. For a `UnitSquareMesh`, `coords[1]` goes 0..1 and the profile direction must match the marker geometry. This is the same assumption made in `forms.py` and is consistent. No new issue here — the note is that the initial phi IC uses `1 - spatial_var` as a proxy for electrode-to-bulk direction, which may be inconsistent with a custom 2D mesh geometry. This is a pre-existing condition inherited from `forms.py` and is out of scope.

**Reclassify as pre-existing, not a new finding.** Withdrawn.

---

## __init__.py Verification

**Result: CLEAN.**

`__init__.py` re-exports all six public symbols from `dispatch.py` (lines 103–110) and all orchestrator types from `grid_per_voltage.py` (lines 122–126). The `__all__` list (lines 129–149) matches the imports exactly. The import of `build_context_logc`, `build_forms_logc`, `set_initial_conditions_logc` from `dispatch.py` (not from `forms_logc.py` directly) is correct and avoids a potential circular import since `dispatch.py` is a thin router.

---

## dispatch.py Verification

**Result: CLEAN (A4-Q re-confirmed).**

- `_get_formulation` (lines 65–71): correctly normalizes to lowercase, defaults to "concentration". The `isinstance(p, dict)` guard at line 68 is redundant (since `_params_dict` always returns a dict) but harmless.
- `build_forms` (lines 89–105): the mismatch check `is_logc_ctx = bool(ctx.get("logc_transform", False))` correctly catches logc ctx + concentration formulation. The reverse case (concentration ctx + logc formulation) would reach `build_forms_logc(ctx, solver_params)` which would call `fd.split(U)` etc. on a standard ctx — this would fail at the `split` call inside `forms_logc` rather than producing a clean error. This is a pre-existing condition, not a regression.
- `set_initial_conditions` (lines 108–115): the `blob=False` no-op for logc is documented and correct.

---

## Final Verdict

| ID | Severity | Status |
|----|----------|--------|
| C6a | minor | Re-confirmed present |
| C12a | minor | Re-confirmed present |
| C6b | question/benign | Re-confirmed present |
| A4-Q | question | Re-confirmed present |
| N1 | minor (NEW) | E_eq_v=0.0 silently falls through to global E_eq |
| Checks 1–8 | clean | All pass |

**No critical issues found.** The post-Boltzmann ctx update order is correct. All orchestrator-interface contract keys (`phi_applied_func`, `z_consts`, `boltzmann_z_scale`, `dt_const`, `bv_rate_exprs`, `bcs`, `U`, `U_prev`) are correctly typed and mutable as expected by the patched `grid_per_voltage.py` orchestrator.
