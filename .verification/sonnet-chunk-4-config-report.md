# Verification Report: config / nondim / mesh / solvers

**Verifier:** claude-sonnet-4-6  
**Date:** 2026-05-05  
**Files reviewed:**
- `Forward/bv_solver/config.py` (334 lines)
- `Forward/bv_solver/nondim.py` (168 lines)
- `Forward/bv_solver/mesh.py` (65 lines)
- `Forward/bv_solver/solvers.py` (21 lines — post-cleanup stub)
- Cross-references: `Nondim/transform.py`, `Nondim/constants.py`, `Forward/bv_solver/dispatch.py`, `Forward/bv_solver/forms_logc.py`, `Forward/bv_solver/forms_logc_muh.py`, `scripts/_bv_common.py`

---

## ISSUE 1 — WARNING — config.py: `concentration` in formulation whitelist and default

**SEVERITY:** warning  
**LOCATION:** `config.py:65`, `config.py:72`, `config.py:119`, `config.py:135`

**DESCRIPTION:**  
`_VALID_FORMULATIONS = ("concentration", "logc", "logc_muh")` still lists `"concentration"` as a valid choice. Per CLAUDE.md ("Live backends are `forms_logc.py` and `forms_logc_muh.py` — concentration backend removed in the May 2026 cleanup"), the concentration backend has been removed. `_default_bv_convergence_cfg()` also defaults to `"formulation": "concentration"`, and `_get_bv_convergence_cfg` defaults to `"concentration"` when the key is absent.

The dispatcher `_resolve_backend()` in `dispatch.py:78-79` handles this silently — it falls through to `"logc"` for any unknown formulation, including `"concentration"`. So a user who omits `formulation` from their config gets `"concentration"` parsed and accepted, `dispatch.py` silently routes it to the `logc` backend anyway, and no error is raised. This creates a latent ambiguity: the stored config claims one formulation but the code executes another.

**EVIDENCE:**
```python
# config.py:65
_VALID_FORMULATIONS = ("concentration", "logc", "logc_muh")

# config.py:119 — default
"formulation": "concentration",

# dispatch.py:71-79 — silent fallthrough
def _resolve_backend(solver_params):
    formulation = _read_formulation(solver_params)  # returns "logc" default
    if formulation == "logc_muh":
        return "logc_muh"
    return "logc"  # concentration silently maps here
```

**RECOMMENDATION:** Remove `"concentration"` from `_VALID_FORMULATIONS`. Change default in `_default_bv_convergence_cfg` and `_get_bv_convergence_cfg` to `"logc"`. Also update `_validate_formulation`'s `None` default from `"concentration"` to `"logc"`.

---

## ISSUE 2 — CRITICAL — forms_logc.py and forms_logc_muh.py: Stale `exponent_clip=50` hardcoded fallback in IC path

**SEVERITY:** critical  
**LOCATION:** `forms_logc.py:702`, `forms_logc_muh.py:800`

**DESCRIPTION:**  
Both IC helper functions that compute the Picard/Gouy-Chapman initial condition have a local fallback:
```python
exponent_clip = float(conv_cfg.get("exponent_clip", 50.0))
```
This is a **secondary read** from `conv_cfg` rather than from the validated config dict. `conv_cfg` is the result of `_get_bv_convergence_cfg(params)`, which already stores `exponent_clip` as a parsed float — so the `.get("exponent_clip", 50.0)` fallback should never activate in practice. However, if `conv_cfg` is passed in a test or helper that constructs it manually and omits the `exponent_clip` key, the IC path silently uses 50.0 (the revoked legacy value) instead of 100.0.

The main build-forms path (`forms_logc.py:251`) reads `conv_cfg["exponent_clip"]` directly (no fallback), so it would raise `KeyError` rather than silently degrade. The IC path has the opposite behavior: it silently uses the wrong value. This is particularly dangerous because the IC sets Newton's starting point for every voltage — a misconfigured starting exponent directly corrupts the first iterate.

**EVIDENCE:**
```python
# forms_logc.py:702 (IC function body, outside main build_forms_logc)
exponent_clip = float(conv_cfg.get("exponent_clip", 50.0))

# forms_logc_muh.py:800 (IC function body)
exponent_clip = float(conv_cfg.get("exponent_clip", 50.0))

# config.py:111 — authoritative default
"exponent_clip": 100.0,

# CLAUDE.md Hard Rule 2:
# "clip=50 PC is fictitious; clip=100 is the only PC-trustworthy setting"
```

**RECOMMENDATION:** Change both fallback literals from `50.0` to `100.0` to match the authoritative default in `_default_bv_convergence_cfg`. Alternatively, remove the fallback entirely (use `conv_cfg["exponent_clip"]`) so any missing key fails loudly the same way the main build path does.

---

## ISSUE 3 — WARNING — forms_logc.py and forms_logc_muh.py: `u_clamp` fallback of 30.0 vs documented 100.0

**SEVERITY:** warning  
**LOCATION:** `forms_logc.py:211`, `forms_logc_muh.py:262`

**DESCRIPTION:**  
Both build-forms functions read:
```python
_U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))
```
The authoritative default in `_default_bv_convergence_cfg` (config.py:118) is `"u_clamp": 100.0`. The forms modules also carry inline documentation (`forms_logc.py:198–202`) saying to "widen to u_clamp=100 for V_RHE > +0.30 V." If `conv_cfg` is somehow produced without the `u_clamp` key, the fallback 30.0 will bind at V_RHE ≈ 0.30 V and distort the PDE coefficient, exactly the failure mode described in the comment. In normal operation `conv_cfg` always carries the key because `_get_bv_convergence_cfg` writes it; but the mismatch is an inconsistency that could surface in tests that pass partial config dicts.

**EVIDENCE:**
```python
# forms_logc.py:211
_U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))

# config.py:118 (authoritative default)
"u_clamp": 100.0,
```

**RECOMMENDATION:** Change the fallback from `30.0` to `100.0` in both files, or use `conv_cfg["u_clamp"]` (no fallback, will KeyError loudly).

---

## ISSUE 4 — WARNING — config.py `_get_bv_cfg`: alpha validation bypassed for list/tuple inputs

**SEVERITY:** warning  
**LOCATION:** `config.py:24-26`

**DESCRIPTION:**  
In `_get_bv_cfg`, when `alpha` is supplied as a list or tuple (per-species alpha values), validation is skipped:
```python
alpha_val = float(alpha) if not isinstance(alpha, (list, tuple)) else None
if alpha_val is not None and not (0.0 < alpha_val <= 1.0):
    raise ValueError(...)
```
Individual per-species alpha values are passed to `_as_list` without being range-checked. A user could pass `alpha=[0.5, 1.5]` and it would be accepted. The reactions path (`_get_bv_reactions_cfg:316-318`) does validate each alpha, so the multi-reaction path is consistent. The single-reaction path (legacy `bv_bc` block) is not.

**EVIDENCE:**
```python
# config.py:24-26
alpha_val = float(alpha) if not isinstance(alpha, (list, tuple)) else None
if alpha_val is not None and not (0.0 < alpha_val <= 1.0):
    raise ValueError(f"alpha must be in (0, 1]; got {alpha_val}")
# list path skips validation entirely
```

**RECOMMENDATION:** After `_as_list(alpha, n_species, "bv_bc.alpha")`, iterate the result and validate each value is in `(0.0, 1.0]`.

---

## ISSUE 5 — WARNING — config.py `_get_bv_reactions_cfg`: species index bounds not validated for cathodic/anodic species

**SEVERITY:** warning  
**LOCATION:** `config.py:285-326`

**DESCRIPTION:**  
`cathodic_species` and `anodic_species` are parsed as `int(cat)` / `int(anod)` and stored directly. There is no check that `cathodic_species` is within `[0, n_species)`. The `cathodic_conc_factors` loop does check its `species` index (`config.py:305-308`), but the primary `cathodic_species` and `anodic_species` indices are unchecked. An out-of-range index will be silently stored and only fail later when UFL constructs `ci[cathodic_species]` at assembly time, producing an obscure `IndexError`.

**EVIDENCE:**
```python
# config.py:325-326 — no bounds check
"cathodic_species": int(cat),
"anodic_species": int(anod) if anod is not None else None,

# config.py:305-308 — cathodic_conc_factors DOES check
if int(sp_idx) < 0 or int(sp_idx) >= n_species:
    raise ValueError(...)
```

**RECOMMENDATION:** Add bounds checks for `cathodic_species` and `anodic_species` (when not None) after parsing.

---

## ISSUE 6 — WARNING — config.py `_get_bv_boltzmann_counterions_cfg`: `z` sign not validated

**SEVERITY:** warning  
**LOCATION:** `config.py:224`

**DESCRIPTION:**  
The docstring says `z` is "counterion charge number (e.g. -1)" implying an anion, but there is no check that `z_val` is negative (or at least nonzero). `z_val = 0` would produce a Boltzmann factor `exp(0 * phi) = 1` — a constant charge density independent of potential, which is physically meaningless and would corrupt the Poisson equation. A positive `z` would be physically a cation, which the Bikerman closure is not designed for (the derivation in the handoff doc is for the specific case of an inert anion). The code should at minimum reject `z_val == 0`.

**EVIDENCE:**
```python
# config.py:224 — no sign/zero check
z_val = int(entry["z"])
```

**RECOMMENDATION:** Add `if z_val == 0: raise ValueError(...)`. Consider also warning if `z_val > 0` since the Bikerman closure was derived for an anion (z < 0).

---

## ISSUE 7 — NOTE — config.py `_get_bv_reactions_cfg`: `k0` not validated as nonnegative

**SEVERITY:** note  
**LOCATION:** `config.py:323`

**DESCRIPTION:**  
`k0` is stored as `float(rxn.get("k0", 1e-5))` with no sign check. A negative k0 would produce a negative flux at the BV boundary, which has no physical meaning. The legacy `_get_bv_cfg` path also does not validate k0. Both paths should enforce k0 >= 0.

**RECOMMENDATION:** Add `if k0 < 0: raise ValueError(...)` after parsing k0 in both `_get_bv_cfg` and `_get_bv_reactions_cfg`.

---

## ISSUE 8 — NOTE — nondim.py: No double-call guard in `_add_bv_scaling_to_transform`

**SEVERITY:** note  
**LOCATION:** `nondim.py:8-101`

**DESCRIPTION:**  
The function starts from `out = dict(scaling)` and adds `bv_k0_model_vals`, `bv_c_ref_model_vals`, etc. If called twice on the same scaling dict (e.g., via an accidentally double-invoked builder), the second call would re-scale already-scaled k0 values. The forms modules call it exactly once in a single code path (guarded by `use_reactions` branching), so in practice this does not occur. There is no idempotency guard (e.g., checking if `"bv_k0_model_vals"` already exists). This is acceptable for the current architecture but is a latent correctness risk if the call graph changes.

**RECOMMENDATION:** Add an assertion or guard at the top: `assert "bv_k0_model_vals" not in scaling` to catch double-call bugs early.

---

## ISSUE 9 — NOTE — nondim.py: `thermal_voltage_v` hardcoded fallback 0.02569 V

**SEVERITY:** note  
**LOCATION:** `nondim.py:30`

**DESCRIPTION:**  
```python
thermal_voltage_v = scaling.get("thermal_voltage_v", 0.02569)
```
The fallback 0.02569 V is RT/F at ~297.8 K (close to 25°C). In practice `thermal_voltage_v` is always present in the scaling dict because `build_model_scaling` computes it from `temperature_k` and stores it. However, the magic number is only correct at one temperature; if someone passes a partial scaling dict with a non-standard temperature but without `thermal_voltage_v`, the Stern nondim formula (`nondim.py:87`) uses this stale value. The number is correctly documented as `0.02569` in CLAUDE.md context, but the fallback should use the constants module for consistency.

**RECOMMENDATION:** Replace with `GAS_CONSTANT * scaling.get("temperature_k", DEFAULT_TEMPERATURE_K) / FARADAY_CONSTANT` or assert the key is present.

---

## ISSUE 10 — NOTE — solvers.py: Stub file — PETSc options live in scripts layer

**SEVERITY:** note  
**LOCATION:** `solvers.py` (entire file, 21 lines)

**DESCRIPTION:**  
After the May 2026 cleanup, `solvers.py` is a stub containing only `_clone_params_with_phi`. The PETSc/Newton solver options are not centralized in this module — they live in `scripts/_bv_common.py` as a plain dict (`DEFAULT_SOLVER_PARAMS`). The scope requirement asked to verify PETSc options correctness here; the options are correct (as verified in `scripts/_bv_common.py`):

```python
"snes_type":                 "newtonls",
"snes_max_it":               200,
"snes_atol":                 1e-7,
"snes_rtol":                 1e-10,
"snes_stol":                 1e-12,
"snes_linesearch_type":      "l2",
"snes_linesearch_maxlambda": 0.5,
"snes_divergence_tolerance": 1e12,
"ksp_type":                  "preonly",
"pc_type":                   "lu",
"pc_factor_mat_solver_type": "mumps",
```

The solver uses SNES Newton + L2 line search, direct LU (MUMPS), appropriate tolerances, and raises `snes_error_if_not_converged` (set in `grid_per_voltage.py:241`) so divergence is caught. This is correct for a stiff PNP problem.

**RECOMMENDATION:** None on correctness. Consider adding a docstring to `solvers.py` pointing to `scripts/_bv_common.py` for the actual PETSc options, so the file is not silently misleading as a scope entry point.

---

## WHAT IS CORRECT

### config.py

1. **Formulation whitelist and validation** — `_validate_formulation` rejects unknown strings, coerces to lowercase, and raises a descriptive error. The only issue is the stale `"concentration"` entry (Issue 1).

2. **Initializer whitelist** — `_VALID_INITIALIZERS = ("linear_phi", "debye_boltzmann")` and `_validate_initializer` are correct. Both known initializers are present; unknown strings are rejected.

3. **`exponent_clip` default in config** — `_default_bv_convergence_cfg` correctly returns `"exponent_clip": 100.0`, consistent with CLAUDE.md Hard Rule 2 and the 2026-05-04 change. `_get_bv_convergence_cfg` also defaults to `100.0` (line 131).

4. **`u_clamp` default in config** — `_default_bv_convergence_cfg` correctly returns `"u_clamp": 100.0`.

5. **Boltzmann counterion steric_mode whitelist** — Only `"ideal"` and `"bikerman"` are accepted. The `synthesised_4sp` mode referenced in CLAUDE.md's IC note is not a `steric_mode` value; it refers to a specific `boltzmann_counterions` entry structure (with `steric_mode='bikerman'`) and is not a whitelist entry. The whitelist is correct.

6. **Bikerman `a_nondim` requirement** — Correctly required when `steric_mode='bikerman'` and validated as positive.

7. **Backward compat** — `_get_bv_cfg`, `_get_bv_convergence_cfg`, and `_get_bv_boltzmann_counterions_cfg` all handle non-dict `params` gracefully, returning defaults or empty lists without raising.

8. **`_get_bv_convergence_cfg` positivity validators** — All four numeric fields (`exponent_clip`, `conc_floor`, `packing_floor`, `u_clamp`) are validated as strictly positive.

9. **`n_electrons` validation** — Correctly validated as a positive integer.

10. **`reversible` typed via `_bool`** — Correct, handles string truthy values.

11. **`stoichiometry` length check** — Correctly enforced against `n_species`.

### nondim.py

1. **`_add_bv_scaling_to_transform` immutability** — Correctly returns a new dict via `out = dict(scaling)` rather than mutating. Immutability is preserved.

2. **`bv_exponent_scale` derivation** — In dimensional mode: `F/(RT)` = 1/V_T, which is the correct Faraday-over-thermal prefactor for the BV exponent `exp(α·F·η/RT)`. In nondim mode: 1.0, correct because `phi` is already in V_T units and `eta_scaled = (V - E_eq) / V_T` is already dimensionless.

3. **`em` (electromigration prefactor) derivation** — In `Nondim/transform.py:396`: `em = (F/(RT)) * potential_scale`. When `potential_scale = V_T = RT/F`, this gives `em = 1.0`, meaning the NP drift term `em * z * phi` is just `z * phi` (dimensionless phi in V_T units). This is dimensionally consistent with the NP equation in V_T-scaled form. The forms modules use this correctly at `forms_logc.py:327`: `drift = em * z[i] * phi`.

4. **Stern capacitance nondim formula** — `stern_model = C_stern * V_T / (F * c_ref * L)` matches the weak-form derivation in the comments. The formula uses `potential_scale` as the proxy for `V_T` (valid when `potential_scale = V_T`).

5. **k0 / c_ref scaling** — Correctly checks `kappa_inputs_dimless` and `concentration_inputs_dimless` flags before dividing by scale factors. The per-reaction scaling in `_add_bv_reactions_scaling_to_transform` mirrors this correctly.

6. **No re-mutation on the output of `dict(scaling)`** — Both functions create `out = dict(scaling)` and append new keys. Existing keys from the base scaling (including `electromigration_prefactor`, `thermal_voltage_v`) are propagated unchanged.

### mesh.py

1. **Power-law grading formula** — `x_i = (i/N)^beta` for `i=0,...,N`. The `fd.IntervalMesh(N, 1.0)` initially places nodes at uniform `i/N`, then `coords[:] = coords[:] ** beta` applies the monotone power-law map. For `beta > 1`: nodes cluster near `x=0` (electrode), the function is strictly monotone increasing on [0,1], starts at 0, ends at 1.0^beta = 1. This is correct.

2. **Element count** — `IntervalMesh(N, 1.0)` produces `N` cells (elements) and `N+1` nodes. The transform does not add or remove nodes. Correct.

3. **Boundary markers on interval mesh** — Firedrake `IntervalMesh` assigns marker 1 to the left endpoint (x=0, electrode) and marker 2 to the right endpoint (x=1, bulk). The docstring correctly documents this.

4. **2D rectangle grading** — `RectangleMesh(Nx, Ny, 1.0, 1.0)` then `coords[:, 1] = coords[:, 1] ** beta`. Only the y-coordinate (normal to electrode) is stretched; x is left uniform. Boundary markers follow Firedrake `RectangleMesh` convention: 1=left, 2=right, 3=bottom (y=0, electrode), 4=top (y=1, bulk). The docstring correctly documents these. The grading clusters nodes near y=0 (bottom, electrode) for `beta > 1`. Correct.

5. **Default parameters** — `beta=2.0` gives quadratic clustering (reasonable for Debye-layer resolution). `N=300` (interval) and `Ny=300` (rectangle) are appropriate for the production stack.

### solvers.py

1. **`_clone_params_with_phi`** — Correctly handles both the `SolverParams` namedtuple path (`with_phi_applied`) and the legacy list path (mutates index 7). The tuple-to-list copy via `list(solver_params)` is correct for an 11-element params structure.

---

## SUMMARY TABLE

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | WARNING | config.py | `"concentration"` retained in formulation whitelist and as default; backend was removed |
| 2 | CRITICAL | forms_logc.py, forms_logc_muh.py | IC-path fallback `exponent_clip=50.0` (revoked legacy value) |
| 3 | WARNING | forms_logc.py, forms_logc_muh.py | `u_clamp` fallback 30.0 vs authoritative 100.0 in config |
| 4 | WARNING | config.py | Per-species alpha list not range-checked in `_get_bv_cfg` |
| 5 | WARNING | config.py | `cathodic_species`/`anodic_species` not bounds-checked vs `n_species` |
| 6 | WARNING | config.py | Boltzmann counterion `z` not validated as nonzero (z=0 is silently accepted) |
| 7 | NOTE | config.py | `k0` not validated as nonnegative in either BV config path |
| 8 | NOTE | nondim.py | No double-call guard in `_add_bv_scaling_to_transform` |
| 9 | NOTE | nondim.py | `thermal_voltage_v` fallback 0.02569 is a magic number |
| 10 | NOTE | solvers.py | Stub file; actual PETSc options in scripts layer (correct there, but file is misleading as entry point) |

Issues 2 and 3 are in the forms layer (outside the strict four-file scope) but are direct consequences of stale fallbacks that should match the config defaults — flagged here because they were found while cross-referencing how the config values are consumed.
