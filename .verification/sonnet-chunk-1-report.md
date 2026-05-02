# Verification Report: Production Forward-Solver Driver and Param Factory
**Verifier:** Claude Sonnet 4.6  
**Date:** 2026-05-02  
**Scope:** `scripts/plot_iv_curve_unified.py`, `scripts/_bv_common.py`, `Forward/params.py`  
**Context files consulted:** `docs/bv_solver_unified_api.md`, `Forward/bv_solver/dispatch.py`, `Forward/bv_solver/forms_logc.py`, `Forward/bv_solver/boltzmann.py`, `Forward/bv_solver/observables.py`, `Forward/bv_solver/mesh.py`, `Forward/bv_solver/config.py`, `Forward/bv_solver/grid_per_voltage.py`

---

## Summary

**No critical issues found.** Several minor and one noteworthy design point are documented below. The production stack (3sp + Boltzmann counterion + log-c + log-rate BV) is correctly wired end-to-end through all three toggle flags. Constants, signs, stoichiometries, and marker conventions are consistent with the Apr 27 writeup and the docs.

---

## Item-by-Item Findings

### A. Sign on the observable
**VERDICT: CORRECT**

`R_j = cathodic - anodic` in `forms_logc.py` (line 383). For cathodic ORR at negative V_RHE: `cathodic >> anodic`, so `R_j > 0`. `_build_bv_observable_form` computes `scale * sum(R_j) * ds(electrode)` where `scale = -I_SCALE`. The assembled integral is positive (positive R_j integrated over electrode boundary), multiplied by -I_SCALE gives a negative current. The script comment "CD < 0 for cathodic ORR currents" is correct.

The peroxide observable `scale * (R_0 - R_1)` is likewise negative when net H2O2 production exceeds R2 consumption, consistent with the cathodic-negative convention.

No issue.

---

### B. I_SCALE numerical correctness
**VERDICT: CORRECT**

`compute_i_scale()` at `_bv_common.py:152`:
```
I_SCALE = 2 × 96485 × 1.9e-9 × 0.5 / 1e-4 × 0.1 = 0.18332 mA/cm²
```
Computed: 0.183321 mA/cm². The formula `n_e * F * D_ref * c_scale / l_ref * 0.1` is correct. The 0.1 factor converts A/m² → mA/cm² (1 A/m² = 0.1 mA/cm²). Constants match.

No issue.

---

### C. THREE_SPECIES_LOGC_BOLTZMANN preset
**VERDICT: CORRECT**

`_bv_common.py:266-278`:
- Species ordering: 0=O₂ (z=0), 1=H₂O₂ (z=0), 2=H⁺ (z=+1). Correct.
- `c0_vals_hat = [1.0, 1e-4, 0.2]`: O₂ at bulk saturation, H₂O₂ at seed (1e-4 nondim ≈ 5×10⁻⁵ mol/m³), H⁺ at 0.2 (= 0.1 mol/m³ / 0.5 mol/m³).
- `H2O2_SEED_NONDIM = 1e-4`: purely a numerical regularizer for `ln(c0_H2O2)` being finite. It does not affect downstream physics because the Dirichlet BC sets `u_i = ln(1e-4)` at the bulk boundary (y=top), and the actual H₂O₂ at the electrode evolves from BV fluxes. The seed only sets the bulk IC; physical H₂O₂ at steady state is governed by stoichiometry and transport. No physics contamination. ✓

Stoichiometry verification:
- `stoichiometry_r1 = [-1, +1, -2]`: R1 (O₂ + 2H⁺ + 2e⁻ → H₂O₂) consumes 1 O₂ (index 0: -1), produces 1 H₂O₂ (index 1: +1), consumes 2 H⁺ (index 2: -2). Matches writeup. ✓
- `stoichiometry_r2 = [0, -1, -2]`: R2 (H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O) leaves O₂ unchanged (0), consumes 1 H₂O₂ (-1), consumes 2 H⁺ (-2). Water is the solvent, not tracked. Matches writeup. ✓

No issue.

---

### D. DEFAULT_CLO4_BOLTZMANN_COUNTERION and bulk electroneutrality
**VERDICT: CORRECT**

`_bv_common.py:430-434`:
```python
DEFAULT_CLO4_BOLTZMANN_COUNTERION = {
    "z": -1,
    "c_bulk_nondim": C_CLO4_HAT,  # = 0.1 / 0.5 = 0.2
    "phi_clamp": 50.0,
}
```

Bulk electroneutrality check:
- H⁺ (z=+1, c_bulk_nondim=0.2): charge contribution = +0.2
- ClO₄⁻ (z=-1, c_bulk_nondim=0.2): charge contribution = -0.2
- Sum = 0.0 ✓

At φ=0 (bulk reference): Boltzmann factor `exp(-z*φ) = exp(0) = 1` for both species. The Boltzmann residual in Poisson contributes `-charge_rhs * z * c_bulk * 1 = -charge_rhs * (-1) * 0.2 = +0.2 * charge_rhs`, balanced by H⁺'s `-charge_rhs * (+1) * 0.2 = -0.2 * charge_rhs`. Electroneutral. ✓

The `phi_clamp=50.0` prevents `exp(-z*phi)` overflow during Newton iteration; at full nondim potential = V_RHE/V_T ≈ 0.6/0.02569 ≈ 23.4, well within the 50.0 clamp. ✓

No issue.

---

### E. make_bv_solver_params wiring
**VERDICT: CORRECT**

Script call (`plot_iv_curve_unified.py:134-144`) passes:
- `formulation="logc"` → routed to `_make_bv_convergence_cfg(formulation="logc")` → `params['bv_convergence']['formulation'] = 'logc'`. ✓
- `log_rate=True` → `params['bv_convergence']['bv_log_rate'] = True`. ✓
- `boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION]` → `_make_bv_bc_cfg(..., boltzmann_counterions=...)` → `params['bv_bc']['boltzmann_counterions'] = [{...}]` (non-empty). ✓
- `k0_hat_r1`, `k0_hat_r2`, `alpha_r1=ALPHA_R1=0.627`, `alpha_r2=ALPHA_R2=0.5` → passed to `_make_bv_bc_cfg`, wired into `reaction_1` and `reaction_2` dicts. ✓
- `E_eq_r1=0.68`, `E_eq_r2=1.78` → passed through `_make_bv_bc_cfg` → `reaction_1["E_eq_v"] = 0.68`, `reaction_2["E_eq_v"] = 1.78`. ✓

Cross-reference: The dispatcher (`dispatch.py:66-71`) reads `params['bv_convergence']['formulation']` via `_get_formulation()` → routes to `build_context_logc` / `build_forms_logc`. The `boltzmann_counterions` key is at `params['bv_bc']['boltzmann_counterions']` (not under `bv_convergence`), which is exactly where `_get_bv_boltzmann_counterions_cfg` (`config.py:157-160`) looks: `bv_raw = params.get("bv_bc", {}); raw = bv_raw.get("boltzmann_counterions", [])`. ✓

No issue.

---

### F. _make_bv_bc_cfg — H⁺ cathodic concentration factor
**VERDICT: CORRECT**

`_bv_common.py:401-408`:
```python
if attach_h:
    h_factor = [{"species": 2, "power": 2, "c_ref_nondim": c_hp_hat}]
    reaction_1["cathodic_conc_factors"] = h_factor
    reaction_2["cathodic_conc_factors"] = [dict(f) for f in h_factor]
```

- `species: 2` = index of H⁺ in the 3-species ordering (0=O₂, 1=H₂O₂, 2=H⁺). ✓
- `power: 2` = both R1 and R2 consume 2 H⁺ per electron pair, so the cathodic H⁺ dependence is `(c_H+_surf / c_H+_bulk)^2`. ✓
- `c_ref_nondim = c_hp_hat = 0.2` = bulk H⁺ concentration (nondim), so the factor is the normalized surface-to-bulk ratio `(c_H+_surf/c_H+_bulk)^2`, consistent with a Tafel/generalized BV formulation with explicit H⁺ activity. ✓
- Both reactions get this factor (`n_species >= 3` auto-detects, and `attach_h=True` for the 3sp preset). ✓

In the log-rate path (`forms_logc.py:332-338`), the factor enters as `power * (ui[sp_idx] - ln(c_ref))` inside the exponent, which is `2 * (u_H+ - ln(0.2))` = `ln((c_H+_surf / 0.2)^2)`. This is mathematically correct for the log-rate form. ✓

No issue.

---

### G. _make_nondim_cfg
**VERDICT: CORRECT**

`_bv_common.py:324-339`:
- `enabled=True`, all `*_inputs_are_dimensionless=True`. ✓
- `diffusivity_scale_m2_s = D_REF = D_O2 = 1.9e-9` m²/s. ✓
- `concentration_scale_mol_m3 = C_SCALE = C_O2 = 0.5` mol/m³. ✓
- `length_scale_m = L_REF = 1e-4` m. ✓
- `potential_scale_v = V_T = R*T/F ≈ 0.025693` V. ✓
- `kappa_scale_m_s = K_SCALE = D_REF / L_REF = 1.9e-5` m/s. ✓
- `time_scale_s = L_REF² / D_REF = (1e-4)² / 1.9e-9 = 5.263` s. ✓

The `*_inputs_are_dimensionless=True` flags tell `Nondim/transform.py` to treat all passed values as already nondimensionalized — consistent with how `_bv_common.py` pre-computes `D_O2_HAT`, `K0_HAT_R1`, etc. ✓

No issue.

---

### H. SolverParams.from_list / with_phi_applied
**VERDICT: CORRECT WITH A DESIGN NOTE**

`Forward/params.py:29`: `@dataclass(frozen=True)` — the class is frozen. ✓

`from_list` (`params.py:126-144`): maps positional indices correctly:
- [0]=n_species, [1]=order, [2]=dt, [3]=t_end, [4]=z_vals, [5]=D_vals, [6]=a_vals, [7]=phi_applied, [8]=c0_vals, [9]=phi0, [10]=solver_options. ✓

`with_phi_applied` (`params.py:170-172`): uses `dataclasses.replace(self, phi_applied=float(phi))` which creates a new frozen instance. ✓

**DESIGN NOTE (minor):** `__setitem__` (`params.py:90-100`) bypasses the frozen constraint via `object.__setattr__`. This is intentional for legacy backward-compat (code that does `deep_copy()` then mutates by index). It is documented in the docstring. The production path (`grid_per_voltage.py:206-207`) correctly uses `with_phi_applied()` for `SolverParams` instances and only falls back to the list-mutation path for plain lists. The `__setitem__` bypass does not create hidden mutation hazards in the production code path because the orchestrator always invokes `_params_with_phi()` which branches on `isinstance(solver_params, SolverParams)`. ✓

No issue.

---

### I. eta_hat=0.0 as placeholder
**VERDICT: CORRECT**

The script passes `eta_hat=0.0` to `make_bv_solver_params`, which stores it as `phi_applied=0.0` in `SolverParams` (index 7). The orchestrator's `_params_with_phi(phi_applied_target)` calls `solver_params.with_phi_applied(float(phi_applied_target))` at each voltage step (`grid_per_voltage.py:206-207`), creating a new `SolverParams` with the correct per-V potential before building context and forms. The initial placeholder is fully overridden. ✓

No issue.

---

### J. dt=0.25, t_end=80.0 in nondim time
**VERDICT: CORRECT AND REASONABLE**

With `time_scale = L_REF²/D_REF = (1e-4)²/1.9e-9 ≈ 5.263 s`:
- `t_end=80` → 421 physical seconds (ample for diffusion-layer SS over 100 µm). ✓
- `dt=0.25` → 1.32 physical seconds (∼one quarter diffusion time). ✓

The orchestrator (`grid_per_voltage.py:247`) ignores `t_end` from `SolverParams` directly; it runs its own SER adaptive-dt loop with `dt_init=0.25` (matching the `SolverParams` value, consistent), `dt_max_ratio=20.0` → `dt_max=5.0` nondim (26.3 s). The `t_end=80.0` from `SolverParams` is read by `forsolve_bv` (used in the legacy direct-call path) but not by the orchestrator's `_make_run_ss`, which controls termination via the `ss_rel_tol`/`ss_abs_tol`/`ss_consec` criteria instead of a hard `t_end`. This is intentional and consistent with the orchestrator design. ✓

No issue.

---

### K. adj.stop_annotating() context
**VERDICT: CORRECT**

`plot_iv_curve_unified.py:89`: `import firedrake.adjoint as adj` is the import; pyadjoint is not activated (no `adj.continue_annotation()` or `from firedrake.adjoint import *`) before the `with adj.stop_annotating():` block. In Firedrake/pyadjoint, annotation is only active if explicitly enabled; the `stop_annotating()` context manager is a belt-and-suspenders guard. ✓

`grid_per_voltage.py` contains no `adj.*` imports. `forms_logc.py` and `boltzmann.py` contain no adjoint tape operations. The callback (`_grab_observables`) runs inside the `with adj.stop_annotating()` block and calls `fd.assemble(...)` — assembly with annotation off produces no tape entries. No tape leaks across voltages. ✓

No issue.

---

### L. V_RHE_GRID → phi_hat_grid units chain
**VERDICT: CORRECT**

`plot_iv_curve_unified.py:172`: `phi_hat_grid = V_RHE_GRID / V_T` where `V_T ≈ 0.025693 V`. This converts physical Volts → dimensionless potential (in units of V_T). ✓

In `forms_logc.py`, `bv_E_eq_model = E_eq_v / V_T` (via `_add_bv_reactions_scaling_to_transform` in `nondim.py`). The overpotential `eta_hat = phi_applied_model − bv_E_eq_model` is dimensionless. The BV exponent `alpha * n_e * eta_hat` is dimensionless. ✓

At `V_RHE = -0.5 V`: `phi_hat = -0.5 / 0.025693 ≈ -19.5` (dimensionless), within the `exponent_clip=50` guard. ✓  
At `V_RHE = +0.6 V`: `phi_hat ≈ +23.4`, within clip for R1 (`E_eq_r1=0.68 V`, `eta ≈ (23.4 - 26.5) ≈ -3.1` → unclipped). For R2 (`E_eq_r2=1.78 V`, `bv_E_eq_r2_model = 1.78/0.025693 ≈ 69.3`), `eta_r2 ≈ 23.4 - 69.3 = -45.9` → clipped to -50, consistent with the memory note that R2 unclips at V_RHE > +0.495 V only from the cathodic side. ✓

No issue.

---

### M. Output paths and CSV/JSON format
**VERDICT: CORRECT**

`OUT_DIR = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)` — absolute path constructed from `_ROOT` (the PNPInverse directory). `os.makedirs(..., exist_ok=True)` is called before any writes. ✓

CSV writing (`plot_iv_curve_unified.py:213-221`):
- `cd_s = "" if r["cd_mA_cm2"] is None else f"{r['cd_mA_cm2']:.8e}"` — NaN values are converted to `None` at row construction (line 207: `None if np.isnan(...) else float(...)`), so the `is None` guard works correctly. ✓
- `z_factor` is written as `f"{r['z_factor']:.4f}"` where `r['z_factor'] = float(z_achieved[i])`. `z_achieved` is initialized to `np.nan` and always overwritten from `result.points[idx].achieved_z_factor` (the orchestrator always returns a result for every index). Python formats `float('nan')` as the string `'nan'` with `:.4f`, which is legal (though not empty-string; a reader expecting a float will parse it as NaN). ✓

JSON writing: `DEFAULT_CLO4_BOLTZMANN_COUNTERION` is a plain dict (serializable). `V_RHE_GRID.tolist()` converts numpy array to a JSON-serializable Python list. ✓

**MINOR NOTE (M1):** The `z_factor` column in the CSV will contain the string `nan` for any voltage where `z_achieved` was never set by the orchestrator loop. In practice the orchestrator always populates `result.points[idx]` for all indices (see `grid_per_voltage.py:405-412`), so this should not occur in normal operation. However, if someone calls the script with an empty result (no points), `z_achieved[idx]` would remain `np.nan` and the loop `for idx, point in result.points.items()` at line 186 would not execute, leaving `z_achieved` all-NaN. The CSV would then contain `nan` in the z_factor column. This is cosmetic and the `method` column would remain `"MISSING"`. No data corruption, but not empty-string like `cd_mA_cm2`. **Severity: minor.**

---

### N. Mesh marker convention
**VERDICT: CORRECT**

`Forward/bv_solver/mesh.py:47-50` (confirmed by grep):
```
1 = left   (x=0, zero-flux)
2 = right  (x=1, zero-flux)
3 = bottom (y=0, electrode)
4 = top    (y=1, bulk)
```

`_make_bv_bc_cfg` defaults: `electrode_marker=3`, `concentration_marker=4`, `ground_marker=4`. ✓

In `forms_logc.py`:
- `bc_phi_electrode = fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker)` → marker 3 = bottom = electrode. ✓
- `bc_phi_ground = fd.DirichletBC(W.sub(n), 0.0, ground_marker)` → marker 4 = top = bulk reference. ✓
- `bc_ui = [...fd.DirichletBC(W.sub(i), ln(c0_i), concentration_marker)...]` → marker 4 = top = bulk. ✓

No issue.

---

## Cross-Reference Checks

### Dispatcher reads params['bv_convergence']['formulation']
**VERIFIED.** `dispatch.py:67-71`:
```python
def _get_formulation(solver_params):
    p = _params_dict(solver_params)
    bv_conv = p.get("bv_convergence", {})
    return str(bv_conv.get("formulation", "concentration")).strip().lower()
```
`_params_dict` handles both `SolverParams` (via `.solver_options`) and plain list (via `[10]`). When `SolverParams` is passed, `hasattr(solver_params, "solver_options")` is True, so `opts = solver_params.solver_options` (the dict built by `make_bv_solver_params`). The key `bv_convergence.formulation = "logc"` is present. Routes correctly to `build_context_logc` / `build_forms_logc`. ✓

### boltzmann_counterions at params['bv_bc']['boltzmann_counterions']
**VERIFIED.** `_make_bv_bc_cfg` puts it at `cfg["boltzmann_counterions"]` inside the `bv_bc` sub-dict (`_bv_common.py:422-423`). `_get_bv_boltzmann_counterions_cfg` reads `params.get("bv_bc", {}).get("boltzmann_counterions", [])` (`config.py:157-160`). Keys match. ✓

### SolverParams.solver_options access patterns
**VERIFIED.** Both access patterns see the same dict:
- `solver_params.solver_options` → the `Dict[str, Any]` field of the dataclass.
- `solver_params[10]` → `to_list()[10]` → same `self.solver_options`. ✓

`grid_per_voltage.py:193` unpacks via tuple destructuring: `n_s, order, dt, t_end, z_v, D_v, a_v, _, c0, phi0, params = solver_params`, which calls `__iter__` → `to_list()` → same dict. ✓

---

## Issues Found

| ID | Severity | Location | Description | Smallest Fix |
|---|---|---|---|---|
| M1 | minor | `plot_iv_curve_unified.py:220` | `z_factor` in CSV formats as the string `"nan"` (not `""`) for missing voltages, unlike `cd_mA_cm2`/`pc_mA_cm2` which use `""`. Inconsistency may confuse CSV readers expecting uniform empty-string for missing values. | Replace `f"{r['z_factor']:.4f}"` with `"" if np.isnan(r['z_factor']) else f"{r['z_factor']:.4f}"`, and similarly at line 279 in the summary print. Not urgent — orchestrator always sets all indices under normal operation. |
| H1 | design note | `Forward/params.py:90-100` | `__setitem__` bypasses frozen via `object.__setattr__`. Documented and safe in the production path, but any new code that `deep_copy()`s and mutates via `[i]=` will silently succeed on a "frozen" object. The production orchestrator does not use this path. | No change needed for the production script. Long-term: annotate or remove `__setitem__` from `SolverParams` once all legacy callers are migrated. |

---

## No Critical or Major Issues

All three production toggles (`formulation="logc"`, `bv_log_rate=True`, `boltzmann_counterions=[ClO4-]`) are correctly wired from the script through `make_bv_solver_params` → `SolverParams.solver_options` → dispatcher → `build_forms_logc` → `add_boltzmann_counterion_residual`. The observable scale (`-I_SCALE`), mesh markers, stoichiometry, electroneutrality, and time-scale choices are all correct.
