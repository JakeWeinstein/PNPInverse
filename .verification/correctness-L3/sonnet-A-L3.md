# Correctness Audit — L3 Pass (Sonnet-A)

**Auditor:** claude-sonnet-4-6  
**Date:** 2026-05-13  
**Scope:** solver_demo_slide15_no_speculative_cs.py, Forward/bv_solver/dispatch.py, mesh.py, nondim.py, scripts/_bv_common.py (targeted)

---

## Issue 1

**SEVERITY:** warning  
**LOCATION:** `scripts/studies/solver_demo_slide15_no_speculative_cs.py:312` (`_stern_bump_ladder`)  
**TRIGGER:** `--stern-final` is passed as a value below `STERN_ANCHOR=0.10` (e.g., `--stern-final 0.05`, `--stern-final 0.0`, or `--stern-final -0.1`).  
**EVIDENCE:**
```python
def _stern_bump_ladder(target: float) -> list[float]:
    if target <= STERN_ANCHOR:        # 0.05 <= 0.10 -> True
        return [float(target)]        # returns [0.05]
```
When `target <= 0.10`, the function returns `[target]` without complaint. The calling code in `_run_one_factor` then iterates this list and calls `set_stern_capacitance_model(ctx_anchor, 0.05)`, which calls `stern_const.assign(0.05 * factor)` — a **downward** bump from the anchor's `C_S=0.10` to the requested value. `set_stern_capacitance_model` only rejects `c_s_f_m2 < 0`, so `0.0` is accepted and sets the Stern coefficient to exactly zero, silently converting the Robin BC to a degenerate zero-strength condition without switching to the Dirichlet path or the `linear_phi` IC appropriate for that limit.

For `target < 0`: `set_stern_capacitance_model` raises `ValueError` ("must be non-negative"), which is caught by the `except Exception` in the bump loop and treated as a normal bump failure, producing a clean early-return from Stage 3. So negative targets don't produce a wrong result — they fail early. The problematic silent path is `0 <= target < 0.10`.

`_stern_bump_ladder` has no guard at the function entry to reject `target < STERN_ANCHOR` when called with a user-supplied `--stern-final`. The bump ladder docstring does not mention this case.

---

## Issue 2

**SEVERITY:** warning  
**LOCATION:** `Forward/bv_solver/dispatch.py:65`, `dispatch.py:115–120` (`set_initial_conditions`)  
**TRIGGER:** Any typo or unrecognised string for `initializer` (e.g., `"debye-boltzmann"` with a hyphen, `"Debye_Boltzmann"` before the `.lower()` normalisation, or genuinely unknown strings like `"blob"`).  
**EVIDENCE:**
```python
def _read_initializer(solver_params: Any) -> str:
    return _read_bv_convergence_field(solver_params, "initializer", "linear_phi")

# In set_initial_conditions:
if initializer == "debye_boltzmann":
    return set_initial_conditions_debye_boltzmann_logc_muh(ctx, solver_params)
return set_initial_conditions_logc_muh(ctx, solver_params)   # silent fallback
```
An unrecognised initializer string silently falls through to `linear_phi` with no warning or log. The `.lower()` normalisation in `_read_bv_convergence_field` handles case, so only genuine typos or unknown strings hit this. For the demo as configured (`initializer="debye_boltzmann"`) this is unreachable, but it is a latent wrong-result path for future callers. The parallel path for `formulation` (unknown -> `"logc"`) is documented as defensive because config validation catches it at parse time; no equivalent validation exists for `initializer`.

---

## Issue 3

**SEVERITY:** note  
**LOCATION:** `scripts/studies/solver_demo_slide15_no_speculative_cs.py:560–563` (`_to_json_list`)  
**TRIGGER:** `_grab`'s `fd.assemble(f_cd)` raises an exception for a **converged** grid point (e.g., observable form assembly fails on a valid-but-pathological field).  
**EVIDENCE:**
```python
def _to_json_list(arr):
    return [
        float(x) if (np.isfinite(x) and converged[i]) else None
        for i, x in enumerate(arr)
    ]
```
`cd_arr` is initialised to `np.nan`. If `_grab` catches an exception, `cd_arr[orig_idx]` remains `nan`. `_to_json_list` then emits `None` for that index even though `converged[i]=True`. The JSON output shows `converged=True` alongside `cd_mA_cm2=None`, which is internally inconsistent and can mislead downstream analysis. The `n_converged` count is unaffected (it counts solver convergence, not observable assembly). The print statement in `_grab`'s exception path does record the failure, so the log is honest, but the JSON is silent about the reason.

---

## Issue 4

**SEVERITY:** note  
**LOCATION:** `scripts/studies/solver_demo_slide15_no_speculative_cs.py:456` (Stage 2 bump loop)  
**TRIGGER:** Always — `set_stern_capacitance_model` is called **outside** the `with adj.stop_annotating():` context manager that wraps only `_last_solver.solve()`.  
**EVIDENCE:**
```python
for cs_target in bump_ladder:
    try:
        set_stern_capacitance_model(ctx_anchor, float(cs_target))  # OUTSIDE stop_annotating
        with adj.stop_annotating():
            ctx_anchor["_last_solver"].solve()                      # INSIDE
```
`set_stern_capacitance_model` calls `stern_const.assign(nondim_value)` on a Firedrake `Constant`. If pyadjoint is in annotating mode (the default when the adjoint tape is alive), this `assign` is recorded on the tape. Since this is a demo (no adjoint tape is started by the calling code), this is harmless overhead at runtime. However, if the pattern is copy-pasted into a context where pyadjoint is active, the tape accumulates Constant reassignments that are not desired. The Stage 1 anchor solve wraps the entire `solve_anchor_with_continuation` call in `stop_annotating` (line 397), but Stage 2 only wraps the `.solve()` call, leaving the metadata update exposed. Correctness impact: none in the demo. Latent hazard: yes.

---

## Issue 5

**SEVERITY:** note  
**LOCATION:** `scripts/studies/solver_demo_slide15_no_speculative_cs.py:732` (`main`)  
**TRIGGER:** If `L_REF` in `_bv_common.py` is ever changed from `1.0e-4`.  
**EVIDENCE:**
```python
domain_height_hat = L_EFF_M / 1.0e-4   # hardcoded L_REF
```
`L_REF = 1.0e-4` is defined in `_bv_common.py` (line 131). The demo hardcodes `1.0e-4` rather than importing `L_REF`. Currently they match so `domain_height_hat = 1.0`, which is the correct production value. If `L_REF` is changed (e.g., for a future L_eff sweep), the demo's `domain_height_hat` would silently disagree with the nondim stack's `l_eff_m` parameter, because `make_bv_solver_params` receives `l_eff_m=L_EFF_M` but the mesh extent is set by `domain_height_hat`. The mismatch would not raise an error — it would produce a physically wrong mesh-to-domain mapping. The `mesh.py` `_validate_domain_height_hat` range check would not catch it since the value is still within `[1e-3, 10.0]`.

---

## Issue 6

**SEVERITY:** note  
**LOCATION:** `scripts/studies/solver_demo_slide15_no_speculative_cs.py:600–609` (`_parse_factor_list`)  
**TRIGGER:** User passes `--factors inf` or any value where `float(tok) = math.inf`.  
**EVIDENCE:**
```python
factors.append(float(tok))   # no finite check
```
`float("inf")` is accepted. `k0_r4e_target = float(K0_HAT_R4E) * math.inf = math.inf`. `PreconvergedAnchor.__post_init__` validates `k0 > 0.0` — `math.inf > 0.0` is `True`, so the anchor builds. `set_reaction_k0_model(ctx, j, math.inf)` calls `bv_k0_funcs[j].assign(math.inf)`, injecting `inf` into the UFL form. The first Newton residual evaluation would produce `nan` throughout, and SNES would immediately fail. So this does not produce a silently wrong result — it crashes loudly at the first Newton solve. The failure would be caught by the `except Exception` in the anchor block and produce a clean `anchor-build-failed` JSON output. Low risk in practice.

---

## Verified Correct

1. **Stage 2 bump failure → Stage 3 not reached.** On any exception in the bump loop, `bump_err` is set and the function returns early at line 464–498. `ctx_anchor["U"]` may hold a partially-converged Newton iterate at the failed Stern value, but `U_post_bump = snapshot_U(ctx_anchor["U"])` is only reached at line 502 (after the `if bump_err is not None: return` guard). No corrupt state leaks into Stage 3.

2. **Cross-factor shared mutable state.** Each factor iteration calls `_make_sp` which creates a fresh `SolverParams` via `make_bv_solver_params`. `with_solver_options` returns `dataclasses.replace(self, solver_options=opts)` — immutable. The shared mesh object has its coordinate data set once before the loop and is never mutated inside `_run_one_factor`. Firedrake `Function` and `Constant` objects created in `build_context`/`build_forms` are per-ctx and not shared. No cross-factor contamination.

3. **A_HP_PHYSICAL derivation.** `_a_nondim_from_radius_m(2.80e-10) = (4/3)*pi*(2.8e-10)^3 * N_A * C_SCALE ≈ 6.645e-5`. This is dimensionally correct (`a_phys [m^3/mol] * C_SCALE [mol/m^3] = dimensionless`). `A_HP_PHYSICAL < A_DEFAULT` (6.6e-5 vs 0.01) because physical H3O+ at r=2.8Å is much smaller than the A_DEFAULT effective sphere at r≈14.9Å. This **increases** the Bikerman saturation ceiling (`c_max = C_SCALE / a_nondim ≈ 18,059 mol/m³`) by 150× relative to `A_DEFAULT`, consistent with CLAUDE.md Hard Rule #7. A_O2 and A_H2O2 values are similarly computed and cross-check against the Cs+ and K+ entries in `_bv_common.py`.

4. **`_grab` callback gating.** In `solve_grid_with_anchor` (line 1107–1113), `per_point_callback` is called only inside `if ok:`. Failed warm walks do not invoke the callback, so `cd_arr` and `pc_arr` retain their `np.nan` fill for those indices and are serialised as `None` by `_to_json_list`. All four call sites in `solve_grid_per_voltage_cold_with_warm_fallback` are similarly gated on `snap is not None`.

5. **`with_solver_options` semantics.** `dataclasses.replace(self, solver_options=opts)` — returns a new frozen `SolverParams`. No mutation. The existing `sp` is unchanged after `sp = sp.with_solver_options(new_opts)`.

6. **`dispatch.py` unknown formulation.** `_resolve_backend` falls through to `"logc"` for any unrecognised formulation string. This is explicitly documented as defensive-only because `_validate_formulation` in `config.py` rejects unknown names at `make_bv_solver_params` build time. For the demo with `formulation="logc_muh"` this path is unreachable.

7. **Mesh grading: Ny=80, β=3.0, D=1.0.** Node positions `y_i = (i/80)^3 * 1.0`. Sum of 80 intervals exactly equals 1.0 (floating-point). First interval `Δy_0 ≈ 1.95×10⁻⁶`, last `Δy_79 ≈ 3.70×10⁻²`. `y[0] = 0.0` is marker 3 (electrode). `y[80] = 1.0` is marker 4 (bulk). Both confirmed against `mesh.py` docstring and multiple independent audits in `.verification/`.

8. **`_factor_label` collisions.** `{1.0: "factor_1", 1e-6: "factor_1e-06", 1e-12: "factor_1e-12", 1e-18: "factor_1e-18"}` — all distinct. Python `format(f, 'g')` uses scientific notation for values below `1e-4`.

9. **`adj.stop_annotating()` lifecycle on exceptions.** Python `with` blocks call `__exit__` on any exit (including exceptions), so annotation state is always restored. The Stage 1 anchor wraps `solve_anchor_with_continuation` completely. Stage 2 wraps `.solve()` per rung but not `set_stern_capacitance_model`. Stage 3 (`solve_grid_with_anchor`) wraps its entire loop in a single `adj.stop_annotating()` context.

10. **`c_ref_nondim` double-scaling.** `make_bv_solver_params` sets `concentration_inputs_are_dimensionless=True` in the nondim config (line 519). In `_add_bv_reactions_scaling_to_transform`, line 150 checks `if nondim_enabled and not concentration_inputs_dimless:` before dividing. Since `concentration_inputs_dimless=True`, `cathodic_conc_factors[]["c_ref_nondim"] = C_HP_HAT` (already dimensionless) is passed through unchanged. No double-scaling.

11. **IC/residual steric consistency.** `build_context_logc_muh` at line 409–412 assigns `a_vals_list[i]` to `steric_a_funcs[i]` — the same values that come from `SpeciesConfig.a_vals_hat = [A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL]`. `build_steric_boltzmann_expressions` uses those same functions. IC Picard and the Newton residual share the same `steric_a_funcs` objects. No inconsistency.

12. **Zero-delta warm walk at anchor point.** `V_RHE_GRID[-1] = 0.55 = ANCHOR_V_RHE`. This grid point is visited first (distance = 0). `warm_walk_phi(v_anchor_eta=V, v_target_eta=V, ...)`: `_march(V, V, 0)` produces `substeps = [V, V, …, V]` (8 identical values). Each `run_ss` call at an already-converged state passes trivially. The final `run_ss` at `v_target_eta` also passes. Returns `True` cleanly.

---

## VERDICT: CONCERNS FOUND

Two warnings (Issue 1, Issue 2) describe silent wrong-result paths reachable through user input. Neither is triggered by the default run (`K0_R4E_FACTORS` standard set, `--stern-final` defaulting to `STERN_BASELINE=0.20`). Four notes describe latent hazards or inconsistencies that do not affect the nominal demo.
