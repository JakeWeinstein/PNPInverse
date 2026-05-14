# Verification Report — Forward Solver Codepath Doc

**Target:** `docs/solver/forward_codepath_demo_slide15.md` audited against current implementation
**Date:** 2026-05-13
**Level:** 2 (Sonnet + Opus in parallel)
**Scope:** 11 files, ~9580 lines, 2 subsystems (`scripts/`, `Forward/bv_solver/`)
**Agents:** Sonnet ×3 + Opus ×3 (6 total)
**Chunking:** 3 chunks per tier — A=entry+scaffolding, B=forms+boltzmann+picard, C=continuation+grid+observables
**Verdict:** **ISSUES FOUND** — substantive but bounded. Doc is structurally faithful but has multiple identifier-level bugs that would mislead anyone writing follow-on code.

---

## Summary

The doc's overall call-graph topology, line-number anchors for function definitions, and convergence-mechanism descriptions are largely correct. However, the doc contains **5 critical identifier/attribution errors** and **~13 secondary inaccuracies** in parameter names, ctx keys, dataclass field names, fallback semantics, and module paths. None of these would cause the *code* to behave differently — but each one would cause a reader writing follow-on code from the doc to use a wrong name and hit a `NameError` / `KeyError` / `AttributeError`.

One doc claim worth highlighting: the demo really does call `ctx['_last_solver'].solve()` in a manual Python loop for Stage 2 Stern bump (verified against `solver_demo_slide15_no_speculative_cs.py:454-458`). Sonnet-C initially flagged this as critical but had confused it with a separate `c_s_ladder` code path inside `solve_anchor_with_continuation` that the demo does not use. After spot-check against source, Opus-C's reading is correct — see "Agreement Analysis" below.

---

## Findings — Critical

| # | Location in doc | Issue | Found by |
|---|-----------------|-------|----------|
| C1 | line 73, line 179 (Layer 1 row) | Function name `solve_reaction_k0_model` does not exist anywhere. Actual function at `anchor_continuation.py:246` is `set_reaction_k0_model` (it's a setter, not a solve routine). | Opus-A, Sonnet-C, Opus-C (3 of 3 in scope) |
| C2 | lines 56-58 (Stage 1 step 2 inside `build_forms_logc_muh`) | `NonlinearVariationalProblem`, `NonlinearVariationalSolver`, and `ctx['_last_solver']` are NOT created inside `build_forms_logc_muh`. They are created in `anchor_continuation.py:1101-1107`, after `build_forms` has returned. Doc misattributes a key piece of Stage 1 orchestration. | Sonnet-B, Opus-B (both) |
| C3 | line 57 | `SNES_OPTS_CHARGED` is a phantom constant — `grep` finds it nowhere in the codebase. Solver options are read from `params['solver_options']` at call time with `snes_error_if_not_converged=True` as the only unconditional default. | Sonnet-B, Opus-B (both) |
| C4 | line 104 (Stage 2 Stern bump) | `ctx['stern_capacitance_func']` is wrong — actual ctx key is `ctx['stern_coeff_const']` (a `fd.Constant`, not a `Function`). The mechanism described (mutate-in-place so residual sees new C_S without rebuild) is correct in spirit; just the key name is wrong. | Sonnet-C, Opus-C (both) |
| C5 | line 48, line 225 (Files-touched table) | `build_model_scaling` is annotated to `nondim.py` and the table puts it under `Forward/bv_solver/nondim.py`. The function actually lives in `Nondim/transform.py:172` (top-level Nondim package). The local `Forward/bv_solver/nondim.py` only contains BV-specific scaling augmenters. | Sonnet-A, Opus-A, Sonnet-B, Opus-B (4 of 4) |

---

## Findings — Warnings (parameter/field name mismatches)

| # | Location in doc | Issue | Found by |
|---|-----------------|-------|----------|
| W1 | lines 20-21 | `_make_sp(stern=0.10, ...)` — actual kwarg is `stern_capacitance_f_m2`, not `stern`. Calling as documented yields `TypeError`. | Sonnet-A |
| W2 | line 23 | Doc lists `THREE_SPECIES_LOGC_BOLTZMANN` as the species preset. Demo does NOT pass this preset; it constructs a fresh `SpeciesConfig` inline with physical Marcus/Stokes `a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL]` (this IS the point of the demo per CLAUDE.md Hard Rule #7). The doc's "physical a_nondim" claim is substantively correct; the preset-name shorthand is misleading because it implies the unmodified preset is passed. | Sonnet-A (critical), Opus-A (warning, with reframing) |
| W3 | line 102 (Stage 2 ladder annotation) | `_stern_bump_ladder(target)` is annotated as `← (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)`, implying 7 rungs every run. For the default production run (`target=0.20`), it returns `[0.20]` — a SINGLE rung. The full 7-tuple is only returned for the no-Stern path (`target=100.0`). | Sonnet-A, Opus-A (both) |
| W4 | line 187 (Layer 9 row) | `adj.stop_annotating()` is described as wrapping demo "solver calls" generally. Demo only wraps Stages 1 (line 397) and 2 (line 457). Stage 3 is wrapped *internally* inside `solve_grid_with_anchor` at `grid_per_voltage.py:1060`, not at the demo level. The doc's framing is technically false for Stage 3. | Sonnet-A, Opus-A (with location nuance) |
| W5 | line 104 | `set_stern_capacitance_model` is described as living "in anchor_continuation.py" with no line number. It's actually at `anchor_continuation.py:448`. Minor — but worth pinning for the doc's pattern. | Sonnet-C, Opus-C |
| W6 | line 108 | `PreconvergedAnchor(phi_eta, ...)` — actual field name is `phi_applied_eta`, not `phi_eta`. | Sonnet-C, Opus-C (both) |
| W7 | line 109 | `PreconvergedAnchor(... dof_count ...)` — actual field name is `mesh_dof_count`. | Opus-C |
| W8 | line 139 (warm_walk_phi signature) | Doc shows `warm_walk_phi(..., bisect_depth_warm=5)`. Inside `warm_walk_phi` the kwarg is named `bisect_depth` (default 3); `bisect_depth_warm` is the *outer* parameter on `solve_grid_with_anchor` (default 5) that gets passed through as `bisect_depth=bisect_depth_warm`. | Sonnet-C, Opus-C (both) |
| W9 | line 43, "build_context_logc_muh" description | Doc says this function constructs `W = V_u × V_μH × V_φ`, `U`, `U_prev`. It actually delegates entirely to `build_context_logc` (in `forms_logc.py`) and only adds `ctx["logc_muh_transform"] = True`. The W/U/U_prev construction is one level down. | Sonnet-B |
| W10 | line 63 ("2×2 scalar Picard on (ψ_S, ψ_D)") | Picard does NOT iterate on (ψ_S, ψ_D) as primary unknowns. It iterates on per-reaction surface rates `R = (R_1, ..., R_N)` (N=2 for the production parallel-2e/4e stack, hence the "2×2" coincidence). ψ_S, ψ_D are *derived* each iter via a 1-D Stern-Robin bisection (single-ion) or linear-Debye match (multi-ion). Also, the actual call goes to `picard_outer_loop_general` (N-reaction), not a `2×2` legacy variant. | Sonnet-B, Opus-B (both) |
| W11 | line 64 ("falls back to linear_phi if Picard oscillates") | Fallback fires on *any* Picard failure path — singular Jacobian, non-finite state, `mu_h_idx_unsupported`, `no_boltzmann_counterion`, n<3, plus oscillation/max-iters. Doc's "if Picard oscillates" is one trigger among many. | Sonnet-B, Opus-B (both) |
| W12 | lines 80-83, line 184 (Layer 6 plateau formula) | Doc shows the plateau predicate as additive `|Δj_cd| < ss_rel_tol·|j_cd| + ss_abs_tol`. The actual code (`grid_per_voltage.py:195-198`) uses OR-of-two-tests with a max-denominator: `is_steady = (Δ/max(\|fv\|,\|prev\|,abs_tol) <= rel_tol) or (Δ <= abs_tol)`. Not algebraically equivalent — the code is more forgiving at small fluxes. | Opus-C |
| W13 | OPUS deeper item I framing | "Consecutive counter resets on failure" — the counter resets when the *plateau test* is False, not when Newton fails (Newton failure exits the closure via `return False`). Semantically close but imprecise. | Opus-C |

---

## Findings — Notes (correct but easy to misread)

| # | Location in doc | Issue | Found by |
|---|-----------------|-------|----------|
| N1 | line 72 (Stage 1 pseudocode) | Pseudocode `for j, k_target in k0_targets:` would fail in Python since `k0_targets` is a `Dict[int, float]`. Real code uses `.items()`. Minor pseudocode issue; the described behavior is correct. | Sonnet-C, Opus-C (both) |
| N2 | line 50 (Tresset annotation) | "Tresset Eq. (19)" attribution exists only in `writeups/May13th/analytic_counterion_derivation.tex` — the `boltzmann.py` source code and docstrings have NO Tresset citation. The math is defensible (the function does implement the Tresset form on the equilibrium subset), but a reader inspecting the code won't find the label. | Sonnet-B, Opus-B |
| N3 | line 50 (steric mention) | The shorthand `set_ic_debye_boltzmann_logc_muh` is consistently used in the doc — actual function is `set_initial_conditions_debye_boltzmann_logc_muh`. Doc-style abbreviation; readers searching for the literal won't find it without the longer prefix. | Opus-A, Opus-B |
| N4 | line 53 (add_boltzmann_counterion_residual mention) | This function is called with `skip_bikerman=True` from `build_forms_logc_muh` (forms_logc_muh.py:881). For the Cs⁺/SO₄²⁻ demo (both are bikerman), it is effectively a no-op. The real Poisson source for these species is built inline at `forms_logc_muh.py:652-660`. The doc places it in the flow as if it does work — for this demo it doesn't. | Opus-B |
| N5 | line 186 (Layer 8, "32× refinement") | 2⁵ = 32 is correct for "times the *failed interval* is halved". But each recursion still uses `n_substeps=8` inside the halved interval, so the *finest substep* is 8·32 = 256× smaller than depth-0. A reader could misread "32× refinement" as the substep ratio. | Opus-C |
| N6 | line 43 (mixed space notation) | "W = V_u × V_μH × V_φ" notation is conceptually correct but the blocks are all n+1 copies of the same scalar CG-`order` space. The μH transform changes only the interpretation of `U.sub(mu_h_idx)`, not the function space itself. | Opus-B |

---

## Agreement Analysis

**Agreement (high confidence):**
- All 4 sub-agents that touched it agree on **build_model_scaling location** (C5). One critical, three warnings.
- Both Chunk-B agents agree on **solver creation misattribution** (C2) and **phantom SNES_OPTS_CHARGED** (C3).
- Both Chunk-C agents agree on **wrong ctx key `stern_capacitance_func`** (C4), **wrong PreconvergedAnchor field name** (W6), and **bisect_depth vs bisect_depth_warm signature mismatch** (W8).
- Opus-A and both Chunk-C agents agree on **solve_reaction_k0_model → set_reaction_k0_model** (C1) — found across two independent chunks.
- Both Chunk-B agents agree on **Picard iterates on R not ψ** (W10) and **fallback fires on any failure** (W11).

**Disagreements:**
1. **Stage 2 Stern bump mechanism (CRITICAL by Sonnet-C, PASS by Opus-C).** Resolved against source: `solver_demo_slide15_no_speculative_cs.py:454-458` literally does `for cs_target in bump_ladder: set_stern_capacitance_model(...); with adj.stop_annotating(): ctx_anchor["_last_solver"].solve()` — exactly as the doc says. Sonnet-C found a *different* code path inside `solve_anchor_with_continuation` (at `anchor_continuation.py:1271-1317`) that triggers when the caller passes a `c_s_ladder` parameter, which *does* run `_run_k0_ladder` per rung. The demo does NOT use that path. **Opus-C correct; Sonnet-C confused the alternative branch with the demo's actual path.** No critical issue here. ❌ Sonnet-C false positive.

2. **Stage 3 adj-wrap location.** Sonnet-A flagged "Stage 3 not wrapped at demo level"; Opus-A added the nuance "Stage 3 IS wrapped internally inside `solve_grid_with_anchor` at `grid_per_voltage.py:1060`". Both true; Opus's framing is more complete. The doc's Layer-9 wording ("demo script around solver calls") is the source of the imprecision.

3. **THREE_SPECIES_LOGC_BOLTZMANN preset (CRITICAL by Sonnet-A, WARNING by Opus-A).** Both agents found the same fact: demo constructs a fresh `SpeciesConfig` with physical radii, overriding the preset. Disagreement is on severity. The doc's "physical a_nondim" claim is **substantively correct**; the preset name shorthand misleads only readers who would grep for `THREE_SPECIES_LOGC_BOLTZMANN`. **Opus-A's warning framing is more accurate** — graded as W2 here.

**Coverage gaps (none material):** All 11 files in scope received at least 2 agent reviews. All 16+ `file:line` annotations in the doc were checked.

---

## Suggested edits (priority order)

1. **C1** — `solve_reaction_k0_model` → `set_reaction_k0_model` (doc lines 73 and 197, plus the files-touched table at line 196)
2. **C2 + C3** — Move "NonlinearVariationalProblem / NonlinearVariationalSolver(options=SNES_OPTS_CHARGED) → ctx['_last_solver']" from inside `build_forms_logc_muh` (line 56-58) to a new step under `solve_anchor_with_continuation`. Replace `SNES_OPTS_CHARGED` with "options from `sp[10]['solver_options']`".
3. **C4** — `ctx['stern_capacitance_func']` → `ctx['stern_coeff_const']` (line 104)
4. **C5** — `build_model_scaling` location: change "(nondim.py)" → "(Nondim/transform.py)" (line 48 and files-touched table at line 225)
5. **W1** — `_make_sp(stern=0.10, ...)` → `_make_sp(stern_capacitance_f_m2=0.10, ...)` (lines 20-21)
6. **W2** — Replace `THREE_SPECIES_LOGC_BOLTZMANN` (line 23) with a note that the demo builds a custom `SpeciesConfig` with physical Marcus/Stokes radii, *overriding* the preset's `A_DEFAULT=0.01`.
7. **W3** — Annotate `_stern_bump_ladder(target)` as returning `[0.20]` for production / `(0.20…100.0)` for no-Stern (line 102).
8. **W6 + W7** — `PreconvergedAnchor` field rename: `phi_eta` → `phi_applied_eta`, `dof_count` → `mesh_dof_count` (lines 108-109).
9. **W8** — In `warm_walk_phi(...)` pseudocode (line 139), use `bisect_depth=5`; clarify `bisect_depth_warm` is the orchestrator kwarg upstream.
10. **W10 + W11** — Rewrite Picard description (lines 62-65) to say "Picard iterates on per-reaction surface rates `R`; ψ_S, ψ_D are derived each iter via Stern-Robin bisection (single-ion) or linear-Debye match (multi-ion). Fallback to muh linear-phi IC fires on any Picard failure (singular Jacobian, non-finite state, mu_h_idx mismatch, oscillation/max-iters)."
11. **W12** — Rewrite plateau formula (lines 80-83) to match the actual OR-of-two-tests: `(Δ/max(|fv|,|prev|,abs_tol) ≤ rel_tol) or (Δ ≤ abs_tol)`.
12. **W4** — Adjust Layer-9 row (line 187) to acknowledge that Stage 3's `adj.stop_annotating()` is *inside* `solve_grid_with_anchor`, not at the demo level.

---

## What the doc gets right (worth preserving)

- Every `file:line` annotation for *function definitions* points to the real definition: `anchor_continuation.py:902`, `:719`, `:246`, `dispatch.py:82/89/96`, `forms_logc_muh.py:152/166/997`, `boltzmann.py:91/272`, `grid_per_voltage.py:135/218/875`, `observables.py:67`. (15 of 16 are exact; line 246 hits the function but with the wrong name attached — see C1.)
- Mesh, grid, anchor voltage, K0_R4e factor set, output path, default kwargs (`initial_scales`, `max_inserts_per_step`, `ic_at_target`) all verified.
- Convergence-mechanism table (Layers 1-8) is correct in spirit for all 8 mechanisms; some specific identifier transcriptions are wrong but the *physics* attribution is accurate.
- `_build_eta_clipped` clips η_scaled BEFORE α·n_e — matches Hard Rule #2 exactly (Sonnet-B and Opus-B both confirmed).
- `exponent_clip = 100` is the production default (forms_logc_muh.py:871, picard_ic.py:1185, config.py:138).
- (1−A_dyn) factor IS applied to both Cs⁺ and SO₄²⁻ via the shared `free_dyn` variable (boltzmann.py:204, 254).
- AdaptiveLadder sqrt-midpoint rule (`midpoint = math.sqrt(prev * scale)`) is correct geometric mean (anchor_continuation.py:884).
- `solve_grid_with_anchor` static-sort + growing-source-pool semantics verified (grid_per_voltage.py:1039-1050).
- `_march` recursive bisection (with deliberate `v_prev_substep` tracking, left-then-right recursion, and outer/inner checkpoint pattern) verified.
- `(n_e/2)·R_j` weighting in `current_density` mode uses `N_ELECTRONS_REF = 2` — this is the I_SCALE electron-count anchor, NOT Mangan/Ruggiero collection efficiency (Opus-C clarified).
- `solve_grid_with_anchor` is the Phase 5γ/6 successor to C+D — coexists with `solve_grid_per_voltage_cold_with_warm_fallback`; doc correctly uses the newer function.

---

## Verdict

**ISSUES FOUND** — 5 critical (4 agreed across tiers, 1 cross-chunk agreement), 13 warnings, 6 notes.

The doc is **useful as a high-level roadmap** and the call-graph topology is sound. The fixes above are mostly s/old/new/ edits in the doc, not code changes. No bug in the implementation was discovered by this audit — only doc-vs-code drift in identifier names, attribution boundaries, and one phantom constant (`SNES_OPTS_CHARGED`) that may once have existed and been refactored away.

Per-chunk reports preserved at:
- `.verification/sonnet-chunk-A-report.md`, `.verification/opus-chunk-A-report.md`
- `.verification/sonnet-chunk-B-report.md`, `.verification/opus-chunk-B-report.md`
- `.verification/sonnet-chunk-C-report.md`, `.verification/opus-chunk-C-report.md`
