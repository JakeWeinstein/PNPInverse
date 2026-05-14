# Chunk A Verification Report — Entry + scaffolding

**Scope:** `scripts/studies/solver_demo_slide15_no_speculative_cs.py`,
`scripts/_bv_common.py`, `Forward/bv_solver/dispatch.py`,
`Forward/bv_solver/mesh.py`, `Forward/bv_solver/nondim.py`.

**Doc under review:** `docs/solver/forward_codepath_demo_slide15.md`

---

## Summary table

| # | Claim | Status |
|---|---|---|
| 1 | `main() → _run_one_factor()` over `K0_R4e ∈ {1, 1e-6, 1e-12, 1e-18}` | PASS |
| 2 | `setup_firedrake_env()` in `scripts/_bv_common.py` | PASS |
| 3 | `make_graded_rectangle_mesh(Nx=8, Ny=80, beta=3.0, domain_height_hat=1.0)` | PASS |
| 4a | `_make_sp(stern=0.10, k0_r4e_factor=f)` → sp_anchor | PASS |
| 4b | `formulation="logc_muh"` | PASS |
| 4c | `species = THREE_SPECIES_LOGC_BOLTZMANN` | **WARNING** — preset NOT passed; demo builds a fresh `SpeciesConfig` with the same topology |
| 4d | physical a_nondim (O2 r=1.7Å, H2O2 r=2.0Å, H+ r=2.8Å) | PASS (demo overrides preset's `A_DEFAULT=0.01`) |
| 4e | Cs⁺/SO₄²⁻ Bikerman counterions | PASS |
| 4f | `exponent_clip = 100` | PASS |
| 5 | `_make_sp(stern=0.20, …)` → sp_baseline | PASS |
| 6 | `_stern_bump_ladder` constant `(0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)` | PASS (constant) — **WARNING** doc presentation suggests this is what the function always returns, but for `target=0.20` it returns `[0.20]` |
| 7a | `dispatch.py:82` = `build_context` | PASS |
| 7b | `dispatch.py:89` = `build_forms` | PASS |
| 7c | `dispatch.py:96` = `set_initial_conditions` | PASS |
| 8 | Stage 3 grid call signature | PASS |
| 9 | 25 V_RHE points, [−0.40, +0.55] V | PASS |
| 10 | Stage 1 anchor at V_RHE=+0.55 V, C_S=0.10 | PASS |
| 11 | Stage 1 ladder `(1e-12, 1e-9, 1e-6, 1e-3, 1.0)`, max_inserts=4, ic_at_target=True | PASS |
| 12 | Output path `StudyResults/solver_demo_slide15_no_speculative_cs/factor_{f:g}/iv_curve.json` | PASS |
| 13 | `adj.stop_annotating()` wrapper around solver calls (demo) | **WARNING** — demo wraps only Stages 1 & 2; Stage 3 relies on internal wrap inside `solve_grid_with_anchor` |
| 14 | `build_model_scaling` in `Forward/bv_solver/nondim.py` | **CRITICAL** — function lives in `Nondim/transform.py`, NOT `Forward/bv_solver/nondim.py` |
| 15 | `make_graded_rectangle_mesh` in `mesh.py` | PASS |
| -- | `solve_reaction_k0_model` (line 73 + line 197 of doc) | **CRITICAL** — actual name is `set_reaction_k0_model`; the doc misnames it in two places |
| -- | `set_ic_debye_boltzmann_logc_muh` (line 61 of doc) | **WARNING** — abbreviation; actual name is `set_initial_conditions_debye_boltzmann_logc_muh` |

---

## Detailed findings

### CRITICAL #1 — Misnamed function `solve_reaction_k0_model`

* **LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:73`, `:197`
* **DESCRIPTION:** Doc calls the per-rung k0 setter `solve_reaction_k0_model`. Actual function is `set_reaction_k0_model` (it sets the model k0 constant; it does not solve anything by itself).
* **EVIDENCE:**
  * `Forward/bv_solver/anchor_continuation.py:246` — `def set_reaction_k0_model(ctx: dict, j: int, k0_model_value: float) -> None:`
  * No `solve_reaction_k0_model` symbol exists in the codebase (grep returns zero hits).
  * Called from inside the ladder at lines 1084, 1092, 1195, 1391 of `anchor_continuation.py`.

### CRITICAL #2 — `build_model_scaling` is in `Nondim/transform.py`, not `Forward/bv_solver/nondim.py`

* **LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:48`, `:225`
* **DESCRIPTION:** Doc lists `nondim.py → build_model_scaling` in the "Files touched" table, implying the function lives in `Forward/bv_solver/nondim.py`. The actual file `Forward/bv_solver/nondim.py` defines BV-specific helpers (`_add_bv_scaling_to_transform`, `_add_bv_reactions_scaling_to_transform`), NOT `build_model_scaling`. The real `build_model_scaling` lives in `Nondim/transform.py:172`.
* **EVIDENCE:**
  * `grep "def build_model_scaling"` → only hit is `Nondim/transform.py:172`.
  * `forms_logc_muh.py:61` imports it: `from Nondim.transform import build_model_scaling, _get_nondim_cfg, _bool`
  * `forms_logc_muh.py:206` calls `base_scaling = build_model_scaling(...)`.
* **FIX:** Either rename the doc reference to `Nondim/transform.py` or note that `Forward/bv_solver/nondim.py` augments — does not define — the model scaling.

### WARNING #3 — `THREE_SPECIES_LOGC_BOLTZMANN` is referenced but not passed

* **LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:23`
* **DESCRIPTION:** Doc says `_make_sp` uses `THREE_SPECIES_LOGC_BOLTZMANN`. The demo does NOT pass the preset; it constructs a fresh `SpeciesConfig` (lines 160–173 of the demo) with the same topology but with physical `a_vals_hat` overriding the preset's `A_DEFAULT=0.01`.
* **EVIDENCE:**
  * `_bv_common.py:373–386` — `THREE_SPECIES_LOGC_BOLTZMANN = SpeciesConfig(..., a_vals_hat=[A_DEFAULT] * 3, ...)`.
  * Demo, line 158–173 builds the species inline and never imports the preset.
  * Demo line 158–159 (comment) is explicit: "Three-species log-c stack with PHYSICAL per-species a_vals_hat (overrides the THREE_SPECIES_LOGC_BOLTZMANN preset's A_DEFAULT=0.01)."
* **NOTE on CLAUDE.md Hard Rule #7:** Hard Rule #7 (project CLAUDE.md) warns the *general* production stack defaults dynamic species to `A_DEFAULT=0.01`. This specific demo does NOT exhibit that bug — it overrides with physical Marcus/Stokes radii. The doc's "physical a_nondim" claim is true for THIS demo, but the juxtaposition of "THREE_SPECIES_LOGC_BOLTZMANN" + "physical a_nondim" in two adjacent bullets is misleading.
* **FIX:** Reword to "THREE_SPECIES_LOGC_BOLTZMANN topology (custom `SpeciesConfig` with physical a_nondim per species, overriding the preset's `A_DEFAULT`)".

### WARNING #4 — `set_ic_debye_boltzmann_logc_muh` is an abbreviation

* **LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:61`
* **DESCRIPTION:** Doc shows `set_ic_debye_boltzmann_logc_muh`; actual function name is `set_initial_conditions_debye_boltzmann_logc_muh`.
* **EVIDENCE:** `forms_logc_muh.py:997` — `def set_initial_conditions_debye_boltzmann_logc_muh(`
* **FIX:** Either spell out the full name or add a "(abbrev. of set_initial_conditions_…)" note.

### WARNING #5 — `_stern_bump_ladder(target)` value is path-dependent

* **LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:102`
* **DESCRIPTION:** Doc shows `for cs in _stern_bump_ladder(target): ← (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)`. The full 7-tuple is the value of `_STERN_BUMP_LADDER_VERIFIED` (demo line 301). The function output depends on `target`:
  * `target=0.20` (production Stern) → returns `[0.20]` — just one rung.
  * `target=100.0` (no-Stern variant) → returns `[0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0]` — all 7 rungs.
* **EVIDENCE:** Demo lines 304–322:
  ```python
  for rung in _STERN_BUMP_LADDER_VERIFIED:
      if rung >= target:
          rungs.append(float(target))
          return rungs
      rungs.append(float(rung))
  ```
* **FIX:** Note that the 7-rung ladder is the worst case (no-Stern, target=100); the production-Stern case (target=0.20) is just a single 0.20 rung.

### WARNING #6 — Stage 3 grid walk is NOT wrapped by the demo's `adj.stop_annotating()`

* **LOCATION:** `docs/solver/forward_codepath_demo_slide15.md:187`
* **DESCRIPTION:** Doc claims `adj.stop_annotating()` wraps "around solver calls" in the demo script. The demo wraps only:
  * Stage 1 — `solve_anchor_with_continuation` (line 397)
  * Stage 2 — `ctx_anchor["_last_solver"].solve()` Stern bumps (line 457)
  * Stage 3 — `solve_grid_with_anchor` (line 543) is **NOT** wrapped at the call site.
* **MITIGATION:** `solve_grid_with_anchor` wraps internally at `Forward/bv_solver/grid_per_voltage.py:1060`, and `solve_anchor_with_continuation` wraps internally at `anchor_continuation.py:1245`. So the effect (no adjoint tape growth) is preserved; the doc's "wrapper" phrasing is slightly inaccurate about where the wrapping lives.
* **EVIDENCE:**
  * `grep -n "stop_annotating" scripts/studies/solver_demo_slide15_no_speculative_cs.py` — exactly 2 hits at lines 397, 457.
  * `grid_per_voltage.py:1060` — `with adj.stop_annotating():` inside `solve_grid_with_anchor`.
* **FIX:** Doc could note that two wraps are at the demo script call sites and the third is inside `solve_grid_with_anchor`.

---

## Items confirmed correct (no issue)

* **`setup_firedrake_env()`** — `_bv_common.py:45`, exact match.
* **Mesh signature** — Demo line 732 computes `domain_height_hat = L_EFF_M / 1.0e-4 = 100e-6 / 1e-4 = 1.0`; matches doc's `domain_height_hat=1.0`. `MESH_NX=8`, `MESH_NY=80`, `MESH_BETA=3.0`.
* **K0_R4E_FACTORS** — `(1.0, 1e-6, 1e-12, 1e-18)` at demo line 82.
* **`formulation = "logc_muh"`** — `FORMULATION = "logc_muh"` (demo line 92), passed via `make_bv_solver_params(... formulation=FORMULATION, ...)` line 225.
* **Cs⁺/SO₄²⁻** — `boltzmann_counterions=[DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC, DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC]` (demo lines 229–232).
* **`exponent_clip = 100`** — explicitly set at demo lines 240–241; matches default in `_make_bv_convergence_cfg` (`_bv_common.py:485`).
* **`STERN_ANCHOR = 0.10`** (demo line 99) and **`STERN_BASELINE = 0.20`** (demo line 100) — both confirmed.
* **dispatch.py routing** — `build_context` (line 82) → `build_context_logc_muh` (forms_logc_muh.py:152); `build_forms` (line 89) → `build_forms_logc_muh` (forms_logc_muh.py:166); `set_initial_conditions` (line 96) routes to `set_initial_conditions_debye_boltzmann_logc_muh` (forms_logc_muh.py:997) when initializer is `debye_boltzmann`.
* **Grid call signature** — `n_substeps_warm=8`, `bisect_depth_warm=5`, `per_point_callback=_grab` confirmed (demo lines 548–550).
* **V_RHE_GRID** — `tuple(np.linspace(-0.40, +0.55, 25).round(4).tolist())` (demo line 79); `ANCHOR_V_RHE = +0.55` (line 80).
* **`anchor_continuation.py:902`** — `solve_anchor_with_continuation` confirmed; `initial_scales` default `(1e-12, 1e-9, 1e-6, 1e-3, 1.0)`, `max_inserts_per_step=4`, `ic_at_target=True` all match demo arguments.
* **`AdaptiveLadder`** at `anchor_continuation.py:719` — confirmed.
* **`make_run_ss` @ grid_per_voltage.py:135** — confirmed.
* **`warm_walk_phi` @ grid_per_voltage.py:218** — confirmed.
* **`solve_grid_with_anchor` @ grid_per_voltage.py:875** — confirmed.
* **`_build_bv_observable_form` @ observables.py:67** — confirmed.
* **`build_steric_boltzmann_expressions` @ boltzmann.py:91** — confirmed.
* **`add_boltzmann_counterion_residual` @ boltzmann.py:272** — confirmed.
* **Output paths** — `StudyResults/solver_demo_slide15_no_speculative_cs/factor_{f:g}/iv_curve.json` (demo lines 696–698, 701, 747–750).
* **`firedrake.adjoint` import** — Demo line 353 `import firedrake.adjoint as adj`. (Doc's "pyadjoint" mention is technically correct since `firedrake.adjoint` is built on pyadjoint.)
* **Re-exports** — `Forward.bv_solver.__init__.py` exports `make_graded_rectangle_mesh` (line 56) and `solve_grid_with_anchor` (line 97); both demo imports work.

---

## Cross-chunk notes

* The doc claims `forms_logc_muh.py:152` = build_context_logc_muh, `:166` = build_forms_logc_muh, `:997` = set_initial_conditions_debye_boltzmann_logc_muh. These all match the actual source (verified via grep). Whether the body of those functions matches the doc's bullet-point descriptions (mixed FE space W = V_u × V_μH × V_φ, Picard 2×2 on ψ_S/ψ_D, etc.) is in Chunk B's scope.
* `_build_eta_clipped` exists at `forms_logc_muh.py:388` (doc only references it without a line number); the η_scaled clip claim is in scope for Chunk B.
* `set_reaction_k0_model` / `set_stern_capacitance_model` / `set_reaction_kw_eff_model` all live in Chunk C territory (`anchor_continuation.py`); the doc's `solve_reaction_k0_model` typo is a cross-chunk concern for both A and C verification.

---

## VERDICT: ISSUES FOUND

**Critical:**
1. Doc names a function `solve_reaction_k0_model` that does not exist — actual name is `set_reaction_k0_model` (appears twice in doc, lines 73 and 197).
2. Doc files-touched table places `build_model_scaling` in `Forward/bv_solver/nondim.py`; the function actually lives in `Nondim/transform.py`. The local `nondim.py` defines BV-specific helpers (`_add_bv_scaling_to_transform`, `_add_bv_reactions_scaling_to_transform`) that augment but do not define the scaling.

**Warnings (clarity / accuracy):**
3. `THREE_SPECIES_LOGC_BOLTZMANN` is referenced as the species config, but the demo builds a fresh `SpeciesConfig` inline that overrides `a_vals_hat`. The "physical a_nondim" claim is correct for the demo (CLAUDE.md Hard Rule #7 does NOT apply here because the demo deliberately overrides A_DEFAULT). Doc should make the override explicit.
4. `set_ic_debye_boltzmann_logc_muh` is an abbreviation of `set_initial_conditions_debye_boltzmann_logc_muh`.
5. `_stern_bump_ladder(target)` returns the verified 7-tuple only when `target ≥ 100`; for the production `target=0.20` it returns just `[0.20]`.
6. Demo's `adj.stop_annotating()` covers Stages 1–2 only; Stage 3 is wrapped inside `solve_grid_with_anchor` (not at the demo's call site).

No off-by-one or other logic bugs observed in the Chunk A scope. All file:line annotations land on the correct function definitions modulo the renames noted above.
