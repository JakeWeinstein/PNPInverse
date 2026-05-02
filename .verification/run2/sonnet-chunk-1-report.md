# Second-Pass Correctness Verification Report
## Scope
- `scripts/plot_iv_curve_unified.py` (287 lines)
- `scripts/_bv_common.py` (635 lines)
- `Forward/params.py` (212 lines)

Context (not verified): `docs/bv_solver_unified_api.md`, `docs/plot_iv_curve_codepath.md`

Orchestrator read as context: `Forward/bv_solver/grid_per_voltage.py` (confirmed the filter logic)
Config reader read as context: `Forward/bv_solver/config.py` (confirmed key consumption)

---

## 1. Re-verification of Prior Findings

### M1 (minor — CONFIRMED)
**LOCATION:** `scripts/plot_iv_curve_unified.py:220`

**DESCRIPTION:** The `z_factor` column is written with `f"{r['z_factor']:.4f}"` unconditionally. When `z_achieved[orig_idx]` was never set (i.e. the orchestrator returned a `"cold-failed"` point with `achieved_z_factor=0.0` and `converged=False`), the array slot is `np.nan` (initialized at line 158), but the code at line 209 does `float(z_achieved[i])`, which converts `np.nan` to the Python float `nan`, and then at line 220 `f"{r['z_factor']:.4f}"` formats it as `"nan"` — the literal string — rather than an empty cell like the `cd_mA_cm2` / `pc_mA_cm2` columns use.

However, note that `z_achieved` IS populated for every point returned by the orchestrator (Phase 1 always sets `achieved_z_factor` in the result, even for failed points — it sets the partial z achieved, which may be 0.0 for z-ramp failures but is never `nan`). The `"missing"` sentinel covers only `method_at_v` for points not in `result.points`, but `result.points` should include every index. So in practice `z_factor` will be a float (possibly 0.0) for all points, not `nan`. The bug exists in principle (if `result.points` were missing an index, `z_achieved[idx]` remains `nan`), but is unlikely to surface given the orchestrator's loop structure.

**SMALLEST FIX:** For consistency with the cd/pc columns, guard the z_factor format:
```python
z_s = "" if np.isnan(r["z_factor"]) else f"{r['z_factor']:.4f}"
f.write(f"{r['V_RHE']:.4f},{cd_s},{pc_s},{z_s},{r['method']}\n")
```

### H1 (design note — CONFIRMED)
**LOCATION:** `Forward/params.py:90-100` (`__setitem__`)

**DESCRIPTION:** `__setitem__` calls `object.__setattr__` to bypass `@dataclass(frozen=True)`. This is intentional (documented in the docstring), but it is a semantic hole: the frozen guarantee is violated for any caller that uses index-assignment (`params[i] = v`). The immutable helpers (`with_phi_applied`, etc.) correctly use `dataclasses.replace` and do not have this problem.

The H1 design note stands. No regression introduced by this pass.

---

## 2. Sign Conventions and I_SCALE Numerical Correctness

### I_SCALE formula
**LOCATION:** `scripts/_bv_common.py:152-165`

```python
I_SCALE = n_electrons * F_CONST * d_ref * c_scale / l_ref * 0.1
```

Numerically: 2 × 96485 C/mol × 1.9e-9 m²/s × 0.5 mol/m³ / 1e-4 m × 0.1 = 0.18332... mA/cm².

The factor 0.1 converts SI (A/m²) → mA/cm²: 1 A/m² = 0.1 mA/cm². This is correct.

The observable is assembled with `scale=-I_SCALE`. The minus sign makes cathodic ORR currents (which integrate to a negative dimensionless flux in the convention where ORR consumes O₂) appear negative in `cd_mA_cm2`, consistent with the comment at line 197-198 and with convention in the writeup. CORRECT.

### Observable labeling: `cd_nondim` / `pc_nondim`
**LOCATION:** `scripts/plot_iv_curve_unified.py:156-201`

Variables named `cd_nondim` / `pc_nondim` are actually in mA/cm² already (the `scale=-I_SCALE` factor is applied inside `_build_bv_observable_form`). Lines 200-201 correctly re-alias them:
```python
cd_ma_cm2 = cd_nondim   # observable already includes -I_SCALE
pc_ma_cm2 = pc_nondim
```
The variable naming is misleading (the `_nondim` suffix implies dimensionless), but the values are correct and the renaming comment is accurate. No numerical error; minor naming confusion only.

---

## 3. THREE_SPECIES_LOGC_BOLTZMANN Preset

**LOCATION:** `scripts/_bv_common.py:266-278`

```python
THREE_SPECIES_LOGC_BOLTZMANN = SpeciesConfig(
    n_species=3,
    z_vals=[0, 0, 1],           # O2 neutral, H2O2 neutral, H+ charge +1
    d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
    a_vals_hat=[A_DEFAULT] * 3,
    c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT],   # 1.0, 1e-4, 0.2
    stoichiometry_r1=[-1, +1, -2],   # O2 consumed, H2O2 produced, 2 H+ consumed
    stoichiometry_r2=[0, -1, -2],    # H2O2 consumed, 2 H+ consumed
    ...
)
```

**Stoichiometries are physically correct** for the ORR reactions at pH 4 (acidic, Mangan2025 scheme): R1 (O₂ + 2H⁺ + 2e⁻ → H₂O₂) has −1 O₂, +1 H₂O₂, −2 H⁺; R2 (H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O) has 0 O₂, −1 H₂O₂, −2 H⁺.

**H2O2_SEED_NONDIM = 1e-4** seeds the log-c initial condition so `ln(c0_H2O2)` is finite. This is the intended design (documented). It is not a physics change to H₂O₂ behavior.

**z_vals=[0, 0, 1]:** ClO₄⁻ is NOT in this list; it is handled via Boltzmann. H⁺ has z=+1. CORRECT.

**D_O2_HAT = D_O2/D_REF = 1.9e-9/1.9e-9 = 1.0.** CORRECT.
**D_H2O2_HAT = 1.6e-9/1.9e-9 ≈ 0.842.** CORRECT.
**D_HP_HAT = 9.311e-9/1.9e-9 ≈ 4.90.** CORRECT.

**C_O2_HAT = 0.5/0.5 = 1.0** (C_SCALE = C_O2 = 0.5 mol/m³). CORRECT.
**C_HP_HAT = 0.1/0.5 = 0.2.** CORRECT.

---

## 4. Boltzmann Counterion Bulk Neutrality

**LOCATION:** `scripts/_bv_common.py:430-434`

```python
DEFAULT_CLO4_BOLTZMANN_COUNTERION = {
    "z": -1,
    "c_bulk_nondim": C_CLO4_HAT,   # = C_CLO4 / C_SCALE = 0.1 / 0.5 = 0.2
    "phi_clamp": 50.0,
}
```

In the bulk at equilibrium (φ = 0): the Boltzmann counterion contributes `z * c_bulk * exp(-z*φ) = (-1) * 0.2 * exp(+1*0) = -0.2` to the charge density. The dynamic H⁺ contributes `+1 * C_HP_HAT = +0.2`. These cancel exactly → bulk electroneutrality holds for the 3sp+Boltzmann system.

`C_HP_HAT = C_HP / C_SCALE = 0.1 / 0.5 = 0.2`
`C_CLO4_HAT = C_CLO4 / C_SCALE = 0.1 / 0.5 = 0.2`

Physical: C_HP = C_CLO4 = 0.1 mol/m³. That is the electroneutrality condition for HClO₄ at pH 4 (10⁻⁴ mol/L = 0.1 mol/m³). CORRECT.

---

## 5. make_bv_solver_params Wiring

**LOCATION:** `scripts/_bv_common.py:441-557`

The factory builds `params` dict as:
```
params = {all SNES key-value pairs from snes_opts}
params["bv_convergence"] = {formulation, bv_log_rate, ...}
params["nondim"] = {...}
params["bv_bc"] = {reactions, boltzmann_counterions, markers, legacy keys}
```

Then `SolverParams.from_list([n_species, 1, dt, t_end, z_vals, d_vals, a_vals, eta_hat, c0, 0.0, params])` stores this dict as `solver_options`.

The orchestrator unpacks it as:
```python
n_s, order, dt, t_end, z_v, D_v, a_v, _, c0, phi0, params = solver_params
```
Index 10 (`params`) is `solver_options` from `SolverParams.to_list()`. CORRECT.

The dispatcher reads `params['bv_convergence']['formulation']`. Since the factory sets it at `params["bv_convergence"]["formulation"] = "logc"`, this is correctly wired.

The dispatcher reads `params['bv_bc']['boltzmann_counterions']` via `_get_bv_boltzmann_counterions_cfg`. Since the factory sets `params["bv_bc"]["boltzmann_counterions"] = [DEFAULT_CLO4_BOLTZMANN_COUNTERION]`, this is correctly wired.

`bv_log_rate` is in `params["bv_convergence"]["bv_log_rate"] = True`. `_get_bv_convergence_cfg` reads it with `_bool(raw.get("bv_log_rate", False))`. CORRECT.

---

## 6. _make_bv_bc_cfg H+ Factor

**LOCATION:** `scripts/_bv_common.py:399-408`

```python
if include_h_factor is None:
    attach_h = species.n_species >= 3
else:
    attach_h = bool(include_h_factor)
if attach_h:
    h_factor = [{"species": 2, "power": 2, "c_ref_nondim": c_hp_hat}]
    reaction_1["cathodic_conc_factors"] = h_factor
    reaction_2["cathodic_conc_factors"] = [dict(f) for f in h_factor]
```

For `THREE_SPECIES_LOGC_BOLTZMANN` (n_species=3): `attach_h = True`. Species index 2 is H⁺ in both TWO_SPECIES_NEUTRAL, THREE_SPECIES_LOGC_BOLTZMANN (z_vals=[0,0,1]), and FOUR_SPECIES_CHARGED (z_vals=[0,0,1,-1]). Power=2 is correct for 2H⁺ consumed per reaction. `c_ref_nondim = C_HP_HAT = 0.2` (the bulk H⁺ reference). CORRECT.

**ISSUE FOUND — minor:**
`reaction_2["cathodic_conc_factors"]` is set to a copy of the H+ factor. However, R2 (`H₂O₂ → H₂O`) has `"reversible": False` and `"anodic_species": None`. In the log-rate BV branch (`forms_logc.py` as described in the codepath doc), the irreversible R2 has no anodic term. The cathodic rate formula is:
```
log_cathodic = ln(k0) + u_cat + Σ power*(u_sp - ln c_ref) - α*n_e*η
```
Attaching `(c_H+/c_ref)^2` here is physically appropriate for R2 (H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O), so this is correct.

No error here.

---

## 7. _make_nondim_cfg Scales

**LOCATION:** `scripts/_bv_common.py:324-339`

```python
"diffusivity_scale_m2_s": D_REF,           # 1.9e-9 m²/s
"concentration_scale_mol_m3": C_SCALE,     # 0.5 mol/m³
"length_scale_m": L_REF,                   # 1e-4 m
"potential_scale_v": V_T,                  # ~0.025693 V
"kappa_scale_m_s": K_SCALE,                # D_REF/L_REF = 1.9e-5 m/s
"time_scale_s": L_REF**2 / D_REF,         # (1e-4)²/(1.9e-9) ≈ 5.26 s
```

All scales are consistent with the dimensionless system. The flag `*_inputs_are_dimensionless=True` for all quantities means the nondim transform is a pass-through (values go in dimensionless, come out dimensionless). CORRECT.

---

## 8. SolverParams.from_list / with_phi_applied

**LOCATION:** `Forward/params.py:126-144, 170-172`

`from_list` validates `len(params) == 11` and maps positionally. CORRECT.

`with_phi_applied` uses `dataclasses.replace(self, phi_applied=float(phi))` — produces a new frozen instance. CORRECT.

The orchestrator calls `solver_params.with_phi_applied(float(phi_applied_target))` at `grid_per_voltage.py:207` when `isinstance(solver_params, SolverParams)` is True (it will be, since `make_bv_solver_params` returns a `SolverParams`). CORRECT.

---

## 9. eta_hat=0.0 Placeholder

**LOCATION:** `scripts/plot_iv_curve_unified.py:135`

`make_bv_solver_params(eta_hat=0.0, ...)` places `phi_applied=0.0` in `SolverParams` (index 7). The orchestrator immediately overrides this with `_params_with_phi(phi_applied_target)` for each voltage. The 0.0 placeholder is never used as an actual boundary condition. CORRECT.

---

## 10. dt=0.25/t_end=80.0 Sanity

**LOCATION:** `scripts/plot_iv_curve_unified.py:135`

`dt=0.25` (dimensionless time step) and `t_end=80.0` (dimensionless end time). The orchestrator uses `dt_init=0.25` from `grid_per_voltage.py` default (matched by `dt_init` in the factory), and caps via SER. `t_end` is embedded in `SolverParams` but is not directly used by the SS loop in the C+D orchestrator (which runs up to `max_ss_steps` iterations, not to `t_end`); it is present for completeness/backward compat. No issue.

Time scale `L_REF²/D_REF ≈ 5.26 s`, so dt_dim = 0.25 × 5.26 ≈ 1.3 s — reasonable for diffusion-limited system at 100 µm. CORRECT.

---

## 11. adj.stop_annotating()

**LOCATION:** `scripts/plot_iv_curve_unified.py:174`

The entire `solve_grid_per_voltage_cold_with_warm_fallback` call is wrapped in `with adj.stop_annotating()`. This prevents pyadjoint from recording any tape during the forward solve, which is correct for a pure forward evaluation (no gradient needed). CORRECT.

---

## 12. V_RHE_GRID / phi_hat_grid Units

**LOCATION:** `scripts/plot_iv_curve_unified.py:46-49, 172`

```python
V_RHE_GRID = np.array([-0.50, -0.40, ..., 0.60])   # V vs RHE
phi_hat_grid = V_RHE_GRID / V_T                      # dimensionless
```

`V_T ≈ 0.025693 V`. The grid spans from `-0.50/0.025693 ≈ -19.5` to `0.60/0.025693 ≈ 23.3` in dimensionless units. These are passed as `phi_applied_values` to the orchestrator. The orchestrator uses them directly as `phi_applied` (not `eta`), and `use_eta_in_bv=True` means the actual BV overpotential is computed inside the forms as `eta = phi_applied - E_eq_v`. CORRECT.

Note: the orchestrator's docstring parameter is called `phi_applied_values` and the codepath doc (section 1e) confirms the orchestrator builds `_build_for_voltage(phi_applied_target)`, which feeds `phi_applied_func` directly — consistent with the Dirichlet BC convention. CORRECT.

---

## 13. Output Paths

**LOCATION:** `scripts/plot_iv_curve_unified.py:96-97`

```python
OUT_DIR = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
os.makedirs(OUT_DIR, exist_ok=True)
```

`_ROOT` is set at module level as `os.path.dirname(_THIS_DIR)` where `_THIS_DIR` is the `scripts/` directory. So `_ROOT` = PNPInverse root. The output goes to `PNPInverse/StudyResults/iv_curve_unified/`. CORRECT.

---

## 14. Mesh Markers

**LOCATION:** `scripts/_bv_common.py:410-421`; codepath doc section 1c

`make_graded_rectangle_mesh` uses `fd.RectangleMesh` whose default Firedrake convention for a unit square is:
- Marker 1 = left (x=0)
- Marker 2 = right (x=1)
- Marker 3 = bottom (y=0) → electrode
- Marker 4 = top (y=1) → bulk

`make_bv_solver_params` defaults: `electrode_marker=3, concentration_marker=4, ground_marker=4`. These match the mesh. CORRECT.

---

## 15. Cross-trace: Orchestrator Key Filtering (NEW this pass)

**LOCATION:** `Forward/bv_solver/grid_per_voltage.py:231-235`

```python
_NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
solve_opts = {
    k: v for k, v in (params or {}).items()
    if isinstance(params, dict) and k not in _NON_PETSC_KEYS
}
solve_opts.setdefault("snes_error_if_not_converged", True)
```

The params dict produced by `make_bv_solver_params` contains the following top-level keys:
- From `snes_opts` (after the script's update): `snes_type`, `snes_max_it` (400), `snes_atol` (1e-7), `snes_rtol` (1e-10), `snes_stol` (1e-12), `snes_linesearch_type` ("l2"), `snes_linesearch_maxlambda` (0.3), `snes_divergence_tolerance` (1e10), `ksp_type`, `pc_type`, `pc_factor_mat_solver_type`, `mat_mumps_icntl_8`, `mat_mumps_icntl_14`
- `"bv_convergence"` — FILTERED OUT (correct)
- `"nondim"` — FILTERED OUT (correct)
- `"bv_bc"` — FILTERED OUT (correct)

`"robin_bc"` is listed in `_NON_PETSC_KEYS` but is never added by `make_bv_solver_params` — harmless (the filter simply won't match it).

All surviving keys are standard PETSc/SNES keys. VERIFIED CLEAN.

---

## 16. Script's snes_opts.update vs _NON_PETSC_KEYS (NEW this pass)

**LOCATION:** `scripts/plot_iv_curve_unified.py:124-132`

```python
snes_opts = {**SNES_OPTS_CHARGED}
snes_opts.update({
    "snes_max_it":               400,
    "snes_atol":                 1e-7,
    "snes_rtol":                 1e-10,
    "snes_stol":                 1e-12,
    "snes_linesearch_type":      "l2",
    "snes_linesearch_maxlambda": 0.3,
    "snes_divergence_tolerance": 1e10,
})
```

All 7 overridden keys are standard PETSc SNES parameters. None are in `_NON_PETSC_KEYS`. No spurious non-PETSc key is introduced. VERIFIED CLEAN.

Note: `SNES_OPTS_CHARGED` starts with `snes_max_it=300` (from the base `SNES_OPTS` dict overridden to 300), and the script further overrides it to 400. This is intentional (tighter + more iterations for the production stack). CORRECT.

---

## 17. snes_error_if_not_converged — Not in SNES_OPTS or SNES_OPTS_CHARGED (NEW this pass)

**LOCATION:** `scripts/_bv_common.py:179-199`

`SNES_OPTS` and `SNES_OPTS_CHARGED` do NOT contain `"snes_error_if_not_converged"`. The orchestrator adds it via `solve_opts.setdefault("snes_error_if_not_converged", True)` only after the filter. The script does NOT add it to `snes_opts` either. Therefore:
- The key is injected once by the orchestrator, after filtering.
- It will NOT be double-filtered (it's not in the initial `params` dict, so the filter never sees it).
- It will always be set to `True` (the default via `setdefault`) because no caller overrides it.

VERIFIED CLEAN. The `setdefault` behavior is correct: the orchestrator needs SNES to raise on non-convergence so the `run_ss` try/except loop can detect divergent Newton iterates. The script correctly delegates this to the orchestrator.

---

## 18. packing_floor Key Gap (NEW this pass — minor)

**LOCATION:** `scripts/_bv_common.py:309-321` (`_make_bv_convergence_cfg`); `Forward/bv_solver/config.py:104-106` (`_get_bv_convergence_cfg`)

`_make_bv_convergence_cfg` does NOT write a `"packing_floor"` key into the output dict. `_get_bv_convergence_cfg` reads it with:
```python
packing_floor = float(raw.get("packing_floor", 1e-8))
```
The default is `1e-8`, which is used silently when the key is absent. This is safe because the `logc` backend uses `u_clamp` (written: 30.0) instead of `packing_floor` for concentration bounding. However, `_default_bv_convergence_cfg()` does include `"packing_floor": 1e-8` for completeness. The omission in `_make_bv_convergence_cfg` is a minor inconsistency — not a runtime bug (the default fires correctly). The config parser and the factory are slightly out of sync on which keys they consider "part of the surface."

**SEVERITY:** minor
**SMALLEST FIX:** Add `"packing_floor": 1e-8` to the dict returned by `_make_bv_convergence_cfg`, for consistency with `_default_bv_convergence_cfg`.

---

## 19. c_ref_legacy=[1.0, 0.0, 1.0] in THREE_SPECIES_LOGC_BOLTZMANN (NEW this pass — question)

**LOCATION:** `scripts/_bv_common.py:277`

```python
c_ref_legacy=[1.0, 0.0, 1.0],
```

For H₂O₂ (species 1), `c_ref_legacy=0.0`. This is the per-species legacy path value (read by `_get_bv_cfg` from `bv_bc.c_ref`). In the multi-reaction path (`_get_bv_reactions_cfg` → `reactions`), `c_ref` per reaction is specified separately:
- R1: `"c_ref": 1.0` (O₂ reference)
- R2: `"c_ref": 0.0` (H₂O₂ reference, initial concentration is zero)

These are written in `_make_bv_bc_cfg` (line 376, 387). The legacy path `c_ref_legacy` is only used if the forms module falls through to `_get_bv_cfg` (which reads `bv_bc.c_ref`). Since `bv_bc.reactions` is populated (it is, by `_make_bv_bc_cfg`), the `_get_bv_reactions_cfg` path is always taken in the log-c backend, making `c_ref_legacy` irrelevant. The value 0.0 for H₂O₂ in the legacy path is correct for its usage context.

**VERDICT:** No bug. The legacy fields are harmless placeholders.

---

## 20. stoichiometry_legacy in THREE_SPECIES_LOGC_BOLTZMANN (NEW this pass — minor)

**LOCATION:** `scripts/_bv_common.py:276`

```python
stoichiometry_legacy=[-1, -1, -1],
```

This per-species legacy list is read by `_get_bv_cfg` when the multi-reaction path is not available. Its values (−1, −1, −1 for all three species) are physically meaningless as a combined stoichiometry for both reactions simultaneously — but as documented, the legacy path is only taken for the older concentration backend without the `reactions` sub-dict. Since `_make_bv_bc_cfg` always populates `reactions`, and the log-c backend always uses `_get_bv_reactions_cfg`, this field is never consumed in the production path. No runtime error.

**SEVERITY:** question (documentation gap, not a bug)

---

## 21. FOUR_SPECIES_CHARGED k0_legacy Anomaly (pre-existing, flagged for completeness)

**LOCATION:** `scripts/_bv_common.py:253-254`

```python
k0_legacy=[K0_HAT_R1] * 4,
alpha_legacy=[ALPHA_R1] * 4,
```

For `FOUR_SPECIES_CHARGED`, the legacy path uses `K0_HAT_R1` for all 4 species (not K0_HAT_R2 for species 1, H₂O₂). This would be wrong if the legacy per-species path were active for a 4-species system with two distinct k0s. However, since `_make_bv_bc_cfg` always writes the `reactions` list and the dispatchers always prefer `_get_bv_reactions_cfg` when `reactions` is present, this value is never exercised in production. Not in scope of this run (4sp not used by the script under test), but noted.

---

## 22. E_EQ_R1=0.68, E_EQ_R2=1.78 Passed to Factory (NEW this pass — confirmed correct)

**LOCATION:** `scripts/plot_iv_curve_unified.py:74, 141-143`

`E_eq_r1=0.68 V` (standard potential for O₂/H₂O₂ at pH 4, RHE scale — matches the Mangan2025 writeup).
`E_eq_r2=1.78 V` (standard potential for H₂O₂/H₂O). Both are in V vs RHE. The factory passes them to `_make_bv_bc_cfg`, which places them in `reaction_1["E_eq_v"]` and `reaction_2["E_eq_v"]`. These are then read by `_get_bv_reactions_cfg` → `float(rxn.get("E_eq_v", 0.0))`. CORRECT.

Note: inside `_make_nondim_cfg` (and the downstream `_add_bv_reactions_scaling_to_transform`), the codepath doc says `bv_E_eq_model = E_eq/V_T`. The E_eq values in the config are in physical volts, and the nondim module divides by `V_T`. The `V_T` scale is correctly set in `_make_nondim_cfg` as `"potential_scale_v": V_T`. CORRECT.

---

## 23. V_RHE_GRID Upper Bound: +0.60 V (NEW this pass — question)

**LOCATION:** `scripts/plot_iv_curve_unified.py:46-49`

The grid extends to `+0.60 V`. The memory note `project_unclipping_threshold.md` records "R2 unclips at V_RHE > +0.495 V". With `bv_log_rate=True`, the clip is removed (log-rate form doesn't use the symmetric exponent clip the same way), so the unclipping at V=+0.60 V is not a concern. The comment in the script at line 2 says `V_RHE in [-0.5, +0.6] V`. This is wider than the writeup's validation range `[-0.50, +0.10] V`, but the script is a forward solve at TRUE parameters (not validation of the inverse), so convergence at these anodic voltages depends on the solver. The codepath is correct; convergence is an empirical question.

**VERDICT:** Not a code bug. The script is intentionally broader than the validated inverse window.

---

## Summary of Findings

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| M1 | minor | `plot_iv_curve_unified.py:220` | z_factor formatted as literal `"nan"` instead of `""` when missing (confirmed from run 1; rare in practice) |
| H1 | design note | `Forward/params.py:90-100` | `__setitem__` bypasses `@dataclass(frozen=True)` via `object.__setattr__` (confirmed from run 1; intentional) |
| NEW-1 | minor | `scripts/_bv_common.py:309-321` | `_make_bv_convergence_cfg` omits `"packing_floor"` key; `_get_bv_convergence_cfg` silently uses default 1e-8; minor inconsistency with `_default_bv_convergence_cfg` |

### Items confirmed CLEAN (no new issues found)

- I_SCALE formula and unit conversion: CORRECT
- THREE_SPECIES_LOGC_BOLTZMANN species preset (z_vals, d_vals, c0_vals, stoichiometries): CORRECT
- Boltzmann counterion bulk neutrality (C_HP_HAT = C_CLO4_HAT = 0.2): CORRECT
- make_bv_solver_params param dict key wiring (formulation, bv_log_rate, boltzmann_counterions to correct sub-dict locations): CORRECT
- `_make_bv_bc_cfg` H+ concentration factor (species 2, power 2, c_ref=C_HP_HAT): CORRECT
- `_make_nondim_cfg` scales: CORRECT
- `SolverParams.from_list` positional mapping: CORRECT
- `SolverParams.with_phi_applied` uses `dataclasses.replace`: CORRECT
- `eta_hat=0.0` placeholder always overridden by orchestrator: CORRECT
- `dt=0.25/t_end=80.0` dimensionless sanity: CORRECT
- `adj.stop_annotating()` usage: CORRECT
- `phi_hat_grid = V_RHE_GRID / V_T` unit conversion: CORRECT
- Output paths: CORRECT
- Mesh markers (electrode=3, bulk=4): CORRECT
- Orchestrator key filter (`_NON_PETSC_KEYS`): VERIFIED CLEAN — all top-level SNES keys survive, all sub-dicts are correctly excluded
- Script's `snes_opts.update` keys: all valid PETSc SNES keys, none in `_NON_PETSC_KEYS`
- `snes_error_if_not_converged` not pre-set in factory; correctly injected by orchestrator via `setdefault`: CORRECT
- `E_EQ_R1=0.68`, `E_EQ_R2=1.78` routing to `reaction["E_eq_v"]`: CORRECT

### No critical issues found.

---

*Verifier: Claude Sonnet 4.6 (second-pass)*
*Date: 2026-05-02*
