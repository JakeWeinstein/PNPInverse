# Chunk A Verification Report
**Date:** 2026-05-13  
**Scope:** Entry + scaffolding — `solver_demo_slide15_no_speculative_cs.py`, `scripts/_bv_common.py`, `Forward/bv_solver/dispatch.py`, `Forward/bv_solver/mesh.py`, `Forward/bv_solver/nondim.py`  
**Doc verified:** `docs/solver/forward_codepath_demo_slide15.md`

---

## ISSUES FOUND

---

### ISSUE 1 — CRITICAL
**LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:23`  
**DESCRIPTION:** Doc claims `• THREE_SPECIES_LOGC_BOLTZMANN` is passed to `make_bv_solver_params`. The demo does NOT use the `THREE_SPECIES_LOGC_BOLTZMANN` preset. Instead, `_make_sp` constructs a new `SpeciesConfig` object directly with physical `a_vals_hat` overrides.  
**EVIDENCE:**
- Doc line 23: `• THREE_SPECIES_LOGC_BOLTZMANN`
- Demo line 159 comment: `# Three-species log-c stack with PHYSICAL per-species a_vals_hat (overrides the THREE_SPECIES_LOGC_BOLTZMANN preset's A_DEFAULT=0.01).`
- Demo lines 160–173: `species = SpeciesConfig(n_species=3, ..., a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL], ...)`
- The `THREE_SPECIES_LOGC_BOLTZMANN` preset at `_bv_common.py:373-386` uses `a_vals_hat=[A_DEFAULT] * 3`, i.e., `A_DEFAULT=0.01` for all three dynamic species.

This is directly related to Hard Rule #7 in CLAUDE.md. The doc says the demo uses `THREE_SPECIES_LOGC_BOLTZMANN`, but the demo intentionally creates a custom `SpeciesConfig` with physical radii to override `A_DEFAULT`. The doc claim is factually incorrect about what object is passed.

---

### ISSUE 2 — WARNING
**LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:20-21`  
**DESCRIPTION:** Doc shows `_make_sp(stern=0.10, k0_r4e_factor=f)` as the call signature. The actual keyword argument is `stern_capacitance_f_m2`, not `stern`.  
**EVIDENCE:**
- Demo line 126-131: `def _make_sp(*, stern_capacitance_f_m2, k0_r4e_factor: float, initializer: str = INITIALIZER):`
- Call at demo line 380: `sp_anchor_cs, _ = _make_sp(stern_capacitance_f_m2=STERN_ANCHOR, k0_r4e_factor=factor, ...)`
- Call at demo line 375: `sp_baseline, k0_targets = _make_sp(stern_capacitance_f_m2=stern_final_v, k0_r4e_factor=factor, ...)`

The doc uses `stern=0.10` and `stern=0.20` as shorthands but these are not valid parameter names and will mislead anyone trying to call the function.

---

### ISSUE 3 — WARNING
**LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:102` and the "Files touched" table entry for `nondim.py`  
**DESCRIPTION:** Doc annotates `build_model_scaling` as living in `nondim.py`. The scope list specifies `Forward/bv_solver/nondim.py`. That file does NOT contain `build_model_scaling`. The function is in `Nondim/transform.py` and is imported by `forms_logc_muh.py` as `from Nondim.transform import build_model_scaling`.  
**EVIDENCE:**
- `grep -rn "def build_model_scaling" Forward/bv_solver/nondim.py` → no output
- `grep -rn "def build_model_scaling" Nondim/transform.py` → `Nondim/transform.py:172:def build_model_scaling(`
- `forms_logc_muh.py:61`: `from Nondim.transform import build_model_scaling, _get_nondim_cfg, _bool`
- `Forward/bv_solver/nondim.py` contains only BV-specific scaling helpers (`_add_bv_scaling_to_transform`, `_add_bv_reactions_scaling_to_transform`), not `build_model_scaling`.

The doc's "Files touched" table lists `nondim.py → build_model_scaling` which readers will naturally map to `Forward/bv_solver/nondim.py` (the in-scope file). The actual source is in the top-level `Nondim/` package.

---

### ISSUE 4 — WARNING
**LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:102` (Stage 2 diagram annotation)  
**DESCRIPTION:** Doc annotates `_stern_bump_ladder(target)` with the comment `← (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)`, implying the function always returns all 7 rungs. For the production Stern run (default `target=STERN_BASELINE=0.20`), the function returns only `[0.20]` — a single rung.  
**EVIDENCE:**
- `_STERN_BUMP_LADDER_VERIFIED = (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)` (demo line 301) — this is the constant, not the return value
- `_stern_bump_ladder(0.20)` → `[0.2]` (verified via local Python)
- `_stern_bump_ladder(100.0)` → `[0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]` (the no-Stern path returns the full tuple)

The full tuple is only returned when `target=100.0` (no-Stern path). For the default Stern production run, Stage 2 is just a single Newton solve at C_S=0.20, not 7 rungs. The doc's annotation conflates the verified ladder constant with the returned list.

---

### ISSUE 5 — WARNING
**LOCATION:** `docs/solver/forward_codepath_demo_slide15.md` Layer 9 (convergence table, line 187)  
**DESCRIPTION:** Doc says `adj.stop_annotating()` wraps "solver calls" (plural), implying all three stages are wrapped. In the actual demo, only Stage 1 (anchor build, line 397) and Stage 2 (Stern bump Newton solve, line 457) are wrapped in `adj.stop_annotating()`. Stage 3's `solve_grid_with_anchor` call (line 543) is NOT wrapped.  
**EVIDENCE:**
- Lines with `adj.stop_annotating()`: 397 (anchor), 457 (Stern bump)
- Line 543: `grid_result = solve_grid_with_anchor(...)` — bare call, no `adj.stop_annotating()` wrapper
- `adj` is imported as `import firedrake.adjoint as adj` at demo line 353

This is a factual inaccuracy about what is wrapped. Whether Stage 3 needs wrapping is a separate correctness question for the grid_per_voltage subsystem, but the doc is wrong about coverage.

---

## Confirmed Claims

**Claim 1 — CONFIRMED:** `main() → _run_one_factor() loops over K0_R4e ∈ {1, 1e-6, 1e-12, 1e-18}`  
Demo line 82: `K0_R4E_FACTORS = (1.0, 1e-6, 1e-12, 1e-18)`. `main()` calls `_run_one_factor(factor, ...)` inside `for factor in factors_to_run` (line 740), where `factors_to_run` defaults to `K0_R4E_FACTORS`. Correct.

**Claim 2 — CONFIRMED:** `setup_firedrake_env()` exists in `scripts/_bv_common.py`  
`_bv_common.py:45`: `def setup_firedrake_env() -> None:`. Also imported and called at demo line 156 and 703-704.

**Claim 3 — CONFIRMED:** `make_graded_rectangle_mesh(Nx=8, Ny=80, beta=3.0, domain_height_hat=1.0)` is the call used  
Demo lines 732-736: `domain_height_hat = L_EFF_M / 1.0e-4` (= 100e-6/1e-4 = 1.0); `make_graded_rectangle_mesh(Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA, domain_height_hat=float(domain_height_hat))` with `MESH_NX=8, MESH_NY=80, MESH_BETA=3.0`. Correct.

**Claim 4 (partial) — CONFIRMED (physical a_nondim part):** The demo DOES use physical `a_nondim` values for O2/H2O2/H+ (O2 r=1.70Å, H2O2 r=2.00Å, H+ r=2.80Å). See demo lines 116-118 and the `SpeciesConfig` construction at lines 160-173. The physical radius computation is correct.  
*(But the doc also incorrectly states `THREE_SPECIES_LOGC_BOLTZMANN` is used — see Issue 1.)*

**Claim 4 (Cs+/SO4²⁻ counterions) — CONFIRMED:** Demo lines 229-232 pass `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` and `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC`.

**Claim 4 (exponent_clip=100) — CONFIRMED:** Demo lines 238-242 patch `bv_convergence["exponent_clip"] = 100.0` via `sp.with_solver_options(new_opts)`. This is set after `make_bv_solver_params` returns, not inside it, but the final `sp` used for solving has `exponent_clip=100`.

**Claim 5 — CONFIRMED:** `_make_sp(stern_capacitance_f_m2=0.20, ...)` builds `sp_baseline`. Demo line 375-379: `sp_baseline, k0_targets = _make_sp(stern_capacitance_f_m2=stern_final_v, ...)` where `stern_final_v=STERN_BASELINE=0.20` in default run.

**Claim 6 — CONFIRMED (with note):** `_STERN_BUMP_LADDER_VERIFIED = (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)` is the literal in the demo (line 301). The constant itself is correct. The doc annotation is misleading about what `_stern_bump_ladder(target)` *returns* for the production target (see Issue 4).

**Claim 7 (dispatch.py line numbers) — CONFIRMED:**  
- `dispatch.py:82` is `def build_context(solver_params, *, mesh=None)` — routes to `build_context_logc_muh`. Correct.
- `dispatch.py:89` is `def build_forms(ctx, solver_params)` — routes to `build_forms_logc_muh`. Correct.
- `dispatch.py:96` is `def set_initial_conditions(ctx, solver_params, *, blob=False)` — routes to `set_ic_debye_boltzmann_logc_muh`. Correct.

**Claim 8 — CONFIRMED:** `solve_grid_with_anchor(..., n_substeps_warm=8, bisect_depth_warm=5, per_point_callback=_grab)` matches demo lines 543-551. Constants: `N_SUBSTEPS_WARM=8`, `BISECT_DEPTH_WARM=5` defined at lines 89-90.

**Claim 9 — CONFIRMED:** V_RHE grid: `np.linspace(-0.40, +0.55, 25)` (demo line 79). 25 points, [−0.40, +0.55] V. Correct.

**Claim 10 — CONFIRMED:** Stage 1 anchor cold Newton at V_RHE=+0.55 V, C_S=0.10. Demo lines 80-81: `ANCHOR_V_RHE=+0.55`; line 99: `STERN_ANCHOR=0.10`; lines 380-385 build `sp_anchor` at `stern_capacitance_f_m2=STERN_ANCHOR` then set phi to `anchor_v_rhe/V_T`.

**Claim 11 — CONFIRMED:** `solve_anchor_with_continuation` called with `initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0), max_inserts_per_step=4, ic_at_target=True`. Demo lines 94-96 define constants; lines 398-404 pass them. Exact match.

**Claim 12 — CONFIRMED:** Output path `StudyResults/solver_demo_slide15_no_speculative_cs/factor_{f:g}/iv_curve.json`. Demo line 701: `out_dir = Path(_ROOT) / "StudyResults" / out_name`; line 747: `out_subdir = out_dir / _factor_label(factor)` where `_factor_label` uses `f"factor_{factor:g}"`; line 749: `open(out_subdir / "iv_curve.json", "w")`. Correct.

**Claim 13 (pyadjoint import path) — CONFIRMED:** `import firedrake.adjoint as adj` at demo line 353. The Stage 1 anchor (line 397) and Stage 2 Stern bump (line 457) ARE wrapped. Stage 3 is not (see Issue 5).

**Claim 14 — PARTIALLY CORRECT (file path ambiguous):** `build_model_scaling` is called inside `build_forms_logc_muh`. It IS in a `nondim`-related file, but specifically `Nondim/transform.py`, not `Forward/bv_solver/nondim.py`. The in-scope `Forward/bv_solver/nondim.py` contains BV-specific scaling utilities only. See Issue 3.

**Claim 15 — CONFIRMED:** `mesh.py` contains `make_graded_rectangle_mesh` at line 68.

---

## Cross-Chunk Flags

The following cross-chunk references were checked on the Chunk A side and appear consistent:

- `forms_logc_muh.py:152` → `build_context_logc_muh` — confirmed.
- `forms_logc_muh.py:166` → `build_forms_logc_muh` — confirmed.
- `forms_logc_muh.py:997` → `set_initial_conditions_debye_boltzmann_logc_muh` — confirmed.
- `set_stern_capacitance_model` is imported from `Forward.bv_solver.anchor_continuation` at demo line 357. Chunk C should verify this function exists there.
- `solve_grid_with_anchor` is imported from `Forward.bv_solver` (line 354). Chunk C should verify `grid_per_voltage.py:875`.

---

## VERDICT: ISSUES FOUND

**Summary of issues by severity:**

| # | Severity | File:Line | Issue |
|---|----------|-----------|-------|
| 1 | CRITICAL | doc:23 | Doc says `THREE_SPECIES_LOGC_BOLTZMANN`; demo constructs custom `SpeciesConfig` with physical radii |
| 2 | WARNING | doc:20-21 | Doc shows `stern=0.10` parameter name; actual kwarg is `stern_capacitance_f_m2` |
| 3 | WARNING | doc:48, files-table | `build_model_scaling` annotated to `nondim.py`; actual source is `Nondim/transform.py` |
| 4 | WARNING | doc:102 | `_stern_bump_ladder` annotation shows full 7-rung tuple; default target=0.20 returns only `[0.20]` |
| 5 | WARNING | doc:187 | Doc says `adj.stop_annotating()` wraps "solver calls"; Stage 3 `solve_grid_with_anchor` is NOT wrapped |
