# Chunk-3 Verification Report
## Scope: config.py, nondim.py (_add_bv_reactions_scaling_to_transform), boltzmann.py

**Verifier:** claude-sonnet-4-6  
**Date:** 2026-05-02  
**Files verified:** Forward/bv_solver/config.py (268 lines), Forward/bv_solver/nondim.py (168 lines, reactions path only), Forward/bv_solver/boltzmann.py (129 lines)  
**Context files consulted:** forms_logc.py, Nondim/transform.py, scripts/_bv_common.py

---

## Summary

No critical bugs found. Two major issues, three minor issues, and several questions are documented below.

---

## A. `_get_bv_cfg` (config.py:14–62)

### A1 — MAJOR: Alpha list case skips range validation

**SEVERITY:** major  
**LOCATION:** config.py:24–26

When `alpha` is a list or tuple (as produced by legacy scripts), `alpha_val` is set to `None` and the range check is skipped entirely. Individual per-element validation is also absent. A caller passing `alpha=[-0.5, 2.0]` would not be caught here. The downstream `_as_list` call at line 54 will broadcast or raise on length mismatch but does not check range.

**SMALLEST FIX:** After line 26, add:
```python
if isinstance(alpha, (list, tuple)):
    for i, av in enumerate(alpha):
        if not (0.0 < float(av) <= 1.0):
            raise ValueError(f"bv_bc.alpha[{i}] must be in (0, 1]; got {float(av)}")
```

This is not triggered on the production path (the reactions path handles alpha per-reaction at line 250–252 with full validation), but the legacy path through `_get_bv_cfg` is exposed.

### A2 — MINOR: Stern validation allows 0 but nondim.py filters with > 0

**SEVERITY:** minor  
**LOCATION:** config.py:47 vs nondim.py:80

`_get_bv_cfg` raises only on `stern_capacitance < 0`, accepting 0. The `_add_bv_scaling_to_transform` at nondim.py:80 then checks `> 0`, treating 0 as "no Stern". `forms_logc.py:215` does the same. This is consistent — 0 is silently treated as "disabled" — but the docstring at config.py:36–44 says "None or 0 (default)" and the error message says "non-negative", so the behavior is intended. No fix required, but could document that 0 == disabled.

### A3 — CONFIRMED CORRECT: Marker defaults (3, 4, 4)

`electrode_marker=3`, `concentration_marker=4`, `ground_marker=4` match `RectangleMesh` convention and the script's usage. The comment at lines 30–31 correctly notes the IntervalMesh override. No issue.

### A4 — CONFIRMED CORRECT: `_make_bv_bc_cfg` → `_get_bv_cfg` flow

`_make_bv_bc_cfg` (scripts/_bv_common.py:410–424) populates the top-level legacy keys (`k0`, `alpha`, `stoichiometry`, `c_ref`, `E_eq_v`, markers). `_get_bv_cfg` reads these as fallbacks with `raw.get(...)`. On the reactions path, `_get_bv_reactions_cfg` is called separately; `_get_bv_cfg` is still called for markers and the Stern capacitance. Parsing is clean.

---

## B. `_validate_formulation` / `_default_bv_convergence_cfg` / `_get_bv_convergence_cfg` (config.py:65–127)

### B1 — MINOR: `_default_bv_convergence_cfg` default `conc_floor=1e-8` differs from `_make_bv_convergence_cfg` default `conc_floor=1e-12`

**SEVERITY:** minor  
**LOCATION:** config.py:84 vs scripts/_bv_common.py:313

`_default_bv_convergence_cfg` returns `conc_floor=1e-8` (line 84). `_make_bv_convergence_cfg` in `_bv_common.py` passes `conc_floor=1e-12` explicitly. When a caller passes a `bv_convergence` dict that omits `conc_floor`, `_get_bv_convergence_cfg` defaults to `1e-8` (line 104). This is a one-order-of-magnitude difference and may matter near onset voltages where concentrations approach zero in the non-logc path. No fix needed on the production logc path (logc uses `exp(u_i)` which is always positive), but inconsistency is a latent risk for future concentration-formulation callers that omit `conc_floor`.

**SMALLEST FIX:** Align the defaults: change config.py:84 to `"conc_floor": 1e-12`, or update the _bv_common default to 1e-8. Both are defensible; pick one and document it.

### B2 — CONFIRMED CORRECT: All keys required by forms_logc.py are present

The set of keys returned by `_get_bv_convergence_cfg` (line 116–127) matches all keys read by forms_logc.py via `conv_cfg.get(...)` and `conv_cfg["..."]`:
- `clip_exponent` ✓ (line 83/117)
- `exponent_clip` ✓ (lines 85/118)
- `regularize_concentration` ✓ (line 85/119)
- `conc_floor` ✓ (line 84/120)
- `use_eta_in_bv` ✓ (line 87/121)
- `packing_floor` ✓ (line 88/122)
- `softplus_regularization` ✓ (line 89/123)
- `bv_log_rate` ✓ (line 90/124)
- `u_clamp` ✓ (line 91/125)
- `formulation` ✓ (line 92/126)

Note: `_make_bv_convergence_cfg` in `_bv_common.py` does not include `packing_floor` or `softplus_regularization` by default (only adds `softplus_regularization` when `softplus=True`). Forms_logc uses `conv_cfg.get("packing_floor", 1e-8)` with a fallback, so this is safe.

### B3 — CONFIRMED CORRECT: `_bool`, `_as_list`, `_pos` coercions

All three are imported from `Nondim/transform.py`. Verified:
- `_bool`: handles bool, str ("true/false/yes/no/on/off/1/0"), rejects None. Correct.
- `_as_list`: broadcasts scalar or validates sequence length. Correct.
- `_pos`: enforces strict positivity. Correct.

---

## C. `_get_bv_boltzmann_counterions_cfg` (config.py:134–197)

### C1 — CONFIRMED CORRECT: Empty/missing path returns []

Lines 155–162 handle `params` not being a dict, `bv_bc` not being a dict, and `raw` being `None/[]/()`. All return `[]` cleanly.

### C2 — CONFIRMED CORRECT: Required keys 'z' and 'c_bulk_nondim' enforced

Lines 173–179 raise `ValueError` for missing keys with clear messages. `phi_clamp` defaults to 50.0.

### C3 — CONFIRMED CORRECT: Sign convention consistency

The config docstring at line 142 states: `z * c_bulk * exp(-z * phi)`. Boltzmann.py line 120–122:
```
F_res -= z_scale * charge_rhs * z_const * c_bulk_const * exp(-z_const * phi_clamped) * w * dx
```
Forms_logc.py line 419 (dynamic species):
```
F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
```
Both are subtracted with the same sign convention. For ClO4⁻ with z=-1, c_bulk=0.2, phi>0 (cathodic):
- `exp(-(-1)*phi) = exp(+phi) > 1` → anion accumulates, contributing `-charge_rhs * (-1) * 0.2 * exp(phi)` = `+charge_rhs * 0.2 * exp(phi)` to F_res
- This reduces the Poisson source positively (anion screens the cation charge), consistent with physics. ✓

### C4 — CONFIRMED CORRECT: `c_bulk_nondim` non-negative check

Line 184: `if c_bulk < 0: raise ValueError`. Correct.

---

## D. `_get_bv_reactions_cfg` (config.py:200–268)

### D1 — CONFIRMED CORRECT: None return for missing/empty reactions

Lines 207–215: returns `None` when reactions is absent, not a list, or empty. Triggers legacy path in forms_logc.py:95. Correct.

### D2 — MINOR: `anodic_species` not range-checked

**SEVERITY:** minor  
**LOCATION:** config.py:260

`cathodic_species` index is range-checked indirectly (via `int(cat)` cast and then `c_surf[cat_idx]` in forms_logc). However, neither `cathodic_species` nor `anodic_species` is explicitly bounds-checked against `[0, n_species)`. The `cathodic_conc_factors.species` IS range-checked (lines 239–243). A caller passing `cathodic_species=5` for a 3-species system would only fail at UFL assembly, not at config parse time.

**SMALLEST FIX:** After line 259, add:
```python
if int(cat) < 0 or int(cat) >= n_species:
    raise ValueError(f"Reaction {j}: cathodic_species {cat} out of range [0, {n_species})")
if anod is not None and (int(anod) < 0 or int(anod) >= n_species):
    raise ValueError(f"Reaction {j}: anodic_species {anod} out of range [0, {n_species})")
```

### D3 — CONFIRMED CORRECT: Production reactions from `_make_bv_bc_cfg` pass validation

R1: `k0=K0_HAT_R1 (~1.26e-5)`, `alpha=ALPHA_R1 (0.627)` ∈ (0,1], `n_electrons=2>0`, `cathodic_species=0`, `anodic_species=1`, `stoichiometry=[-1,+1,-2]` length=3=n_species, `c_ref=1.0`, `cathodic_conc_factors=[{species:2, power:2, c_ref_nondim:0.2}]` — species=2 < 3 ✓.

R2: `k0=K0_HAT_R2 (~5.26e-6)`, `alpha=ALPHA_R2 (0.5)` ∈ (0,1], `n_electrons=2>0`, `cathodic_species=1`, `anodic_species=None`, `stoichiometry=[0,-1,-2]` length=3 ✓, `c_ref=0.0`, `cathodic_conc_factors=[{species:2, power:2, c_ref_nondim:0.2}]` ✓.

All validation passes cleanly.

### D4 — CONFIRMED CORRECT: power cast to int (line 246)

`int(f_cfg.get("power", 1))` is appropriate for stoichiometric coefficients. Used in forms_logc.py:338 as `fd.Constant(float(factor["power"]))` in log-rate mode and directly as Python int in line 366 for non-log-rate. Both uses are consistent.

---

## E. `_add_bv_reactions_scaling_to_transform` (nondim.py:104–168)

### E1 — CONFIRMED CORRECT: `bv_exponent_scale`

- `nondim_enabled=True` → `bv_exponent_scale = 1.0` (line 125). Correct: phi is already in V_T units, so the Tafel exponent `exp(±α·n_e·η̂)` needs no prefactor.
- `nondim_enabled=False` → `bv_exponent_scale = F/(R·T)` (line 122). Correct: the standard Butler-Volmer Tafel form with η in Volts requires `F/(RT) ≈ 38.92 V⁻¹` as the exponential prefactor.

### E2 — CONFIRMED CORRECT: `E_eq_model` for global (top-level) field

- Line 123 (nondim disabled): `E_eq_model = E_eq_v` (in Volts). Correct.
- Line 126 (nondim enabled): `E_eq_model = E_eq_v / potential_scale` (dimensionless). Correct.
- `potential_scale` is read from `scaling.get("potential_scale_v", 1.0)` at line 119. `build_model_scaling` stores this as `"potential_scale_v"` (Nondim/transform.py:423). Key name matches. ✓
- In forms_logc.py:210, `scaling["bv_E_eq_model"]` is read into `E_eq_model_global`. This is the top-level E_eq (from `bv_cfg["E_eq_v"]` = 0.0 in the production config). The per-reaction E_eq values are in `srxn["E_eq_model"]` (nondim.py:144). Forms_logc.py:318 reads `rxn.get("E_eq_model", None)` and builds `eta_j` from it. The global `bv_E_eq_model` = 0.0 in dimensionless units (from `_make_bv_bc_cfg`'s `"E_eq_v": 0.0`), which is only used as a fallback for reactions with `E_eq_model = None or 0.0`. Since production reactions have nonzero `E_eq_v` (defaulting to 0.0 in `_make_bv_bc_cfg` but physically meaningful), the per-reaction path fires correctly.

**QUESTION (not a bug):** For the production pipeline with the provided writeup values E_eq_R1=0.68 V and E_eq_R2=1.78 V, these would need to be passed as `E_eq_r1=0.68, E_eq_r2=1.78` to `make_bv_solver_params`. The defaults in `_make_bv_bc_cfg` are both 0.0, so if the caller does not override, the equilibrium potentials are both 0 (no Nernst correction). This is a usage concern, not a code bug.

### E3 — CONFIRMED CORRECT: `k0_model` (kappa_inputs_dimless=True)

Lines 136–137: `kappa_inputs_dimless=True` → `k0_model = rxn["k0"]` (already dimensionless). `K0_HAT_R1 = K0_PHYS_R1 / K_SCALE = 2.4e-8 / (1.9e-5) ≈ 1.26e-5` is indeed dimensionless. ✓

### E4 — CONFIRMED CORRECT: `c_ref_model` (concentration_inputs_dimless=True)

Lines 140–141: `concentration_inputs_dimless=True` → `c_ref_model = rxn["c_ref"]` (kept as-is). R1 has `c_ref=1.0` (O₂ bulk = C_SCALE/C_SCALE). R2 has `c_ref=0.0` (irreversible; anodic path disabled in forms_logc via `float(rxn["c_ref_model"]) > 1e-30` check at line 349). ✓

### E5 — CONFIRMED CORRECT: `cathodic_conc_factors.c_ref_nondim` scaling

Lines 150–153: when `nondim_enabled=True` and `concentration_inputs_dimless=True`, `sf["c_ref_nondim"] = f_cfg["c_ref_nondim"]` (no scaling). `c_ref_nondim = C_HP_HAT = 0.2` is passed through unchanged. ✓

### E6 — CONFIRMED CORRECT: `bv_stern_capacitance_model = None` in reactions path

Lines 166–167: only sets `bv_stern_capacitance_model = None` if not already in `out`. The forms_logc.py reactions branch (lines 134–148) handles Stern independently and calls `scaling.setdefault("bv_stern_capacitance_model", None)` after. Both are consistent: no Stern → None → `use_stern = False`. ✓

### E7 — MAJOR: Global `bv_E_eq_model` is set to the function-level `E_eq_v` parameter, not per-reaction values

**SEVERITY:** major  
**LOCATION:** nondim.py:165

`out["bv_E_eq_model"] = E_eq_model` where `E_eq_model` is computed from the function argument `E_eq_v` (lines 123 or 126). `forms_logc.py:125` passes `E_eq_v = float(bv_cfg.get("E_eq_v", 0.0))` which is the **top-level legacy** `E_eq_v` field from `bv_cfg` (set to 0.0 by `_make_bv_bc_cfg` at line 417 of `_bv_common.py`).

This means `scaling["bv_E_eq_model"] = 0.0` always on the reactions path, regardless of per-reaction `E_eq_v` values. The global `E_eq_model_global` in `forms_logc.py:210` is therefore always 0.0 in dimensionless units.

**Impact analysis:** The global `eta_clipped` at line 238 is `_build_eta_clipped(E_eq_model_global)` where `E_eq_model_global = 0.0`. However, at line 318–323, each reaction checks its own `rxn.get("E_eq_model", None)` and builds a per-reaction `eta_j` if nonzero. So for reactions with E_eq_v != 0, `eta_j` is correctly built from `srxn["E_eq_model"]`. The global `eta_clipped` (E_eq=0) is only used for reactions where `E_eq_j_val is None or E_eq_j_val == 0.0`.

**The issue:** The comparison at line 319 is `E_eq_j_val != 0.0`. Since `srxn["E_eq_model"] = rxn.get("E_eq_v", 0.0) / potential_scale` (line 144), reactions with `E_eq_v=0.0` correctly map to `E_eq_model=0.0` and fall back to the global path. Reactions with `E_eq_v != 0` get their own `eta_j`. So the per-reaction dispatch IS correct.

The remaining concern is the **floating-point comparison** `E_eq_j_val != 0.0` at forms_logc.py:319. If `E_eq_v` is a very small but nonzero value (e.g., from floating-point division), this check works correctly. If `E_eq_v = 0.0` exactly (as produced by `_make_bv_bc_cfg` for both reactions by default), the `0.0 / potential_scale = 0.0` exactly, so the comparison is safe. No bug in the production config; the float comparison is exact for the default 0.0 case.

**Revised severity:** Downgrade to minor/question. The code is correct for the production config. The only risk is a future caller setting a very small but nonzero `E_eq_v` that should be treated as "no equilibrium correction" — the float comparison would incorrectly build a per-reaction eta for it. Recommend using `abs(E_eq_j_val) < 1e-12` instead of `== 0.0`.

**SMALLEST FIX:** Change forms_logc.py:319:
```python
if E_eq_j_val is not None and abs(E_eq_j_val) > 1e-12:
```

---

## F. `add_boltzmann_counterion_residual` (boltzmann.py:37–129)

### F1 — CONFIRMED CORRECT: Residual sign and physics

The Poisson residual in forms_logc.py:417–419:
```
F_res += eps_coeff * dot(grad(phi), grad(w)) * dx
F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
```
Boltzmann.py:120–122:
```
F_res -= z_scale * charge_rhs * z_const * c_bulk_const * exp(-z_const * phi_clamped) * w * dx
```
Both subtract `charge_rhs * z * c * w * dx`. For the analytic counterion ClO4⁻ (z=-1, c_bulk=0.2):
- `exp(-(-1)*phi) = exp(+phi)` for phi>0 (cathodic electrode, negative phi at electrode in convention)
- Contribution: `-charge_rhs * (-1) * 0.2 * exp(+phi) * w` = `+charge_rhs * 0.2 * exp(+phi) * w`
- In the nondim case `charge_rhs_prefactor = 1.0`, so this adds a positive anion density. This correctly screens the net positive charge from H⁺ accumulation. Physics consistent. ✓

### F2 — CONFIRMED CORRECT: `z_scale` exposed as `ctx['boltzmann_z_scale']`

Line 128: `ctx["boltzmann_z_scale"] = z_scale`. Grid_per_voltage.py:291 reads `ctx.get("boltzmann_z_scale")` and assigns to it. The key name matches exactly. ✓

### F3 — CONFIRMED CORRECT: `phi_clamped` is well-posed

Lines 114–117: symmetric clamp `min(max(phi, -50), +50)`. `fd.min_value`/`fd.max_value` produce valid UFL expressions. The clamp at ±50 (in V_T units) prevents overflow for exp(±50) ≈ 5e21 which is enormous but still representable as float64. Practical constraint: for phi_clamp=50, exp(50) ≈ 5.18e21, which times c_bulk=0.2 gives ~1e21 (nondim). This is physically unrealizable but numerically stable. ✓

### F4 — CONFIRMED CORRECT: `c_bulk_val == 0.0` skip (line 110)

Clean, no issue.

### F5 — CONFIRMED CORRECT: `J_form` rederivation and propagation

Line 126: `ctx["J_form"] = fd.derivative(F_res, U)` overwrites the J_form set at forms_logc.py:446. Grid_per_voltage.py:226 reads `ctx["J_form"]` to construct the `NonlinearVariationalProblem`. The `build_forms` call (forms_logc.py:475) calls `add_boltzmann_counterion_residual(ctx, params)` at the end, before returning `ctx`. Grid_per_voltage.py at line 222–224: `ctx = build_forms(ctx, sp)` then line 225–227 constructs the problem using `ctx["J_form"]`. So the post-Boltzmann J is correctly picked up. ✓

### F6 — CONFIRMED CORRECT: Required ctx keys

Forms_logc.py:448–473 (`ctx.update(...)`) sets `F_res`, `U` (set in build_context_logc line 58 and carried), and `W` (line 58). Boltzmann.py:79–84 checks for all three. The ctx update at forms_logc.py:448 runs before line 475 calls `add_boltzmann_counterion_residual`. ✓

Note: `U` is in `ctx` from `build_context_logc` (line 58: `ctx = {..., "U": U, ...}`). It is then also in `ctx.update(...)` at forms_logc.py:448 (not explicitly, but `U` was already set in ctx from build_context_logc). Forms_logc.py:174: `U = ctx["U"]` — it reads U from context, which was set by build_context_logc. The ctx dict passed to build_forms already contains U. ✓

### F7 — CONFIRMED CORRECT: `ctx['boltzmann_counterions']` consumers

Line 127: `ctx["boltzmann_counterions"] = list(counterions)`. Search across the codebase shows only `grid_charge_continuation.py:451` reads `ctx.get("boltzmann_z_scale", None)` (the z_scale, not the counterions list). No other code reads `ctx["boltzmann_counterions"]` directly. It is diagnostic metadata only. The list is a fresh copy from the parsed config (`list(counterions)`), not a reference to an internal mutable structure. ✓

---

## G. Implicit z=0 invariant

### G1 — CONFIRMED CORRECT: `boltzmann_z_scale=0` zeroes Boltzmann contribution completely

When `z_scale.assign(0.0)`, the entire term `z_scale * charge_rhs * z * c_bulk * exp(-z*phi) * w * dx` becomes zero. The dynamic species' charges are separately zeroed via `ctx["z_consts"][i].assign(z_nominal[i] * 0.0)`. These are independent — no cross-coupling. The `J_form = derivative(F_res, U)` reflects the zeroed contributions because `z_scale` is a Function (not a Constant), so its value is folded into the assembled Jacobian matrix at each Newton iteration. ✓

### G2 — CONFIRMED CORRECT: No other path for analytic counterion charge to enter Poisson

The Boltzmann counterion enters Poisson only through the term added by `add_boltzmann_counterion_residual`. There is no other code path in forms_logc.py that adds the ClO4⁻ charge (it has been removed from the dynamic NP system precisely to be an analytic counterion). The dynamic species loop (forms_logc.py:269–286) covers only n=3 species (O₂, H₂O₂, H⁺) and the Poisson source at line 419 sums over those 3. The Boltzmann counterion contributes separately and only through boltzmann.py. ✓

---

## Cross-Reference Checks

### CR1 — CONFIRMED: `potential_scale_v` key name

nondim.py:119: `scaling.get("potential_scale_v", 1.0)`.  
Nondim/transform.py:423: `"potential_scale_v": potential_scale`.  
Key name matches exactly. ✓

### CR2 — CONFIRMED: forms_logc.py contract with config.py/nondim.py

All keys that forms_logc.py reads from `scaling` after `_add_bv_reactions_scaling_to_transform`:
- `"bv_reactions"` — set at nondim.py:163. ✓
- `"bv_exponent_scale"` — set at nondim.py:164. ✓
- `"bv_E_eq_model"` — set at nondim.py:165. ✓
- `"bv_stern_capacitance_model"` — set at nondim.py:166–167, then potentially overridden by forms_logc.py:133–148. ✓
- `"poisson_coefficient"` — from base_scaling (Nondim/transform.py:401). ✓
- `"charge_rhs_prefactor"` — from base_scaling (Nondim/transform.py:429). ✓
- `"D_model_vals"` — from base_scaling. ✓
- `"dt_model"`, `"phi_applied_model"`, `"phi0_model"`, `"c0_model_vals"` — from base_scaling. ✓
- `"electromigration_prefactor"` — from base_scaling. ✓

### CR3 — CONFIRMED: `boltzmann_z_scale` key exact match

boltzmann.py:128: `ctx["boltzmann_z_scale"] = z_scale`  
grid_per_voltage.py:291: `ctx.get("boltzmann_z_scale")`  
grid_charge_continuation.py:165: `ctx.get("boltzmann_z_scale") is not None`  
Exact key name match across all consumers. ✓

---

## Issue Index

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| A1 | **major** | config.py:24–26 | Alpha list case skips range validation |
| D2 | **major** | config.py:259–260 | `cathodic_species`/`anodic_species` not bounds-checked against n_species |
| A2 | minor | config.py:47 vs nondim.py:80 | Stern value 0 is accepted but silently treated as "disabled" |
| B1 | minor | config.py:84 vs _bv_common.py:313 | `conc_floor` defaults differ (1e-8 vs 1e-12) |
| D2-fix | minor | config.py:260 | anodic_species index not range-checked |
| E7-float | question | forms_logc.py:319 | `E_eq_j_val != 0.0` exact float comparison (safe for production, fragile in general) |

No critical bugs found. The production 3sp+Boltzmann+logc+log-rate stack (as wired by `plot_iv_curve_unified.py` / `make_bv_solver_params` with `THREE_SPECIES_LOGC_BOLTZMANN`) passes all validation paths correctly. The two "major" issues (A1, D2) are validation gaps that do not affect the production config but would silently pass malformed inputs.
