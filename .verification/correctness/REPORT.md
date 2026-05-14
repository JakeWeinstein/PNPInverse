# Code-Correctness Audit — Forward Solver Codepath

**Target:** Codepath exercised by `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
**Date:** 2026-05-13
**Level:** 2 (Sonnet ×3 + Opus ×3 in parallel)
**Scope:** 11 files, ~9580 lines, partitioned into 3 chunks
**Mode:** Independent code-correctness review (NOT doc-vs-code; doc was ignored)
**Verdict:** **PASS** with latent concerns — **zero critical correctness bugs found in the codepath as exercised by this demo.**

---

## Bottom line

The implementation is defensively written: explicit `f_lo * f_hi > 0` bracket guards on bisection, `|det| < 1e-300` checks before Jacobian inversion, `snes_error_if_not_converged=True` so divergence raises rather than silently leaves partial state, frozen `SolverParams` so cross-iteration state cannot bleed, fresh ctx/forms/solver per voltage in Stage 3 so cross-stage `_last_solver` aliasing is impossible. The numerical conventions audited (η-clip ordering, Bikerman shared-θ closure math, μ_H NP flux reduction, observable sign chain) check out. Mesh grading sums to 1.0 FP-exact for the production parameters. Physical `a_nondim(H+) = 6.65e-5` matches the radius derivation per Hard Rule #7 to two sig figs.

The 5 findings that survive triage are all **latent risks that the demo's exercised inputs don't trigger** — they would matter for refactors or off-production configurations.

---

## Findings table

| # | Severity in this demo | Item | Location | Found by | Status |
|---|----------------------|------|----------|----------|--------|
| **F1** | **Latent (high if triggered)** | If `sp[10]` (PETSc solver options) is missing required keys (`ksp_type`, `pc_type`), PETSc silently defaults to GMRES+ILU which diverges on production PNP Jacobians. No assertion. In this demo, `make_bv_solver_params` always populates the keys, so the path is unreachable. | `anchor_continuation.py:1095-1106` | Sonnet-C | Latent — production input is well-formed. |
| **F2** | **Medium (downstream-corrected)** | `solve_stern_split` bisection: if the initial bracket has same-sign endpoints (`f_lo * f_hi > 0`), code explicitly falls back to a linear-Debye closed-form approximation. The fallback does NOT satisfy the Stern Robin identity exactly — it's a Picard IC seed. The full solver downstream enforces the Robin BC, so Newton convergence absorbs the off-identity seed. | `picard_ic.py:~259-268` | Sonnet-B (medium), Opus-B (clean) | Disagreement on severity (see below). Real but downstream-corrected. |
| **F3** | **Latent (benign here)** | `warm_walk_phi`'s final-SS run_ss has no `_restore_U` on failure — U/paf left at partially-converged state at `v_target_eta`. Benign in Stage 3 because each grid point builds a fresh ctx (Opus-C verified). Footgun for any future caller that reuses ctx after a `warm_walk_phi` failure. | `grid_per_voltage.py:309-310` | Sonnet-C | Latent — Stage 3 builds fresh ctx per voltage. |
| **F4** | **Latent (numerical)** | `free_dyn = max(1 - A_dyn, 1e-10)` floor zeros Jacobian rows for Boltzmann counterions in saturated regions (`A_dyn → 1`). Algebraic limit is correct (`c_steric → 1e-10/a_k`, sensible cap). May slow Newton in deep-cathodic saturation. Not triggered at production V band; relevant if physical-a-bridge runs push H⁺ steric to its real scale. | `boltzmann.py:~204, 254` | Sonnet-B, Opus-B (both) | Algebraic limit correct; latent numerical risk. |
| **F5** | **Latent (high if triggered)** | log-rate BV branch silently treats negative `k0_model` as disabled (R_j = 0) rather than raising. If a scaling refactor introduces a sign flip in k0 propagation, the solver would silently return zero current instead of erroring. In this demo k0 is always positive (k0_targets × k0_scale, both > 0). | `forms_logc_muh.py` log-rate branch | Sonnet-B | Latent — production k0 always positive. |
| **F6** | **Note** | `_to_json_list` coerces `inf` to `None` on grid points that are flagged as converged. If a converged solution produces an `inf` observable (overflow in BV rate at extreme V), the JSON would show `cd_mA_cm2 = null` alongside `converged[i] = true`, masking solver overflow as "missing data". | demo `_to_json_list` | Opus-A | Cosmetic for production V band; matters at extreme V. |
| **F7** | **Note** | `_stern_bump_ladder(target)` accepts `target < STERN_ANCHOR` (e.g., user passes `--stern-final 0.05`) and returns `[target]`, silently down-bumping below the anchor build's C_S. The `--stern-final` CLI arg has no lower-bound validation. Production runs always use 0.20 or 100. | demo `_stern_bump_ladder` + CLI | Sonnet-A | Input-validation gap. |
| **F8** | **Note** | `dispatch._resolve_backend` silently falls back to `"logc"` for unrecognized `formulation` values; `initializer` falls back to `"linear_phi"`. Both rely on upstream `config._validate_formulation` to reject typos. If validation is bypassed (programmatic sp construction), wrong backend silently selected. | `dispatch.py` | Sonnet-A, Opus-A | Defense-in-depth gap; not currently exploitable. |
| **F9** | **Note** | `nondim.py` BV scaling augmenters (`_add_bv_scaling_to_transform`, `_add_bv_reactions_scaling_to_transform`) — verify idempotence if called twice. Not exercised twice in this codepath. | `nondim.py` | Both A agents looked, no issue surfaced. | No active issue. |
| **F10** | **Note** | Two `fd.derivative(F_res, U)` calls — one in `build_forms_logc_muh:839`, one inside `add_boltzmann_counterion_residual:386`. The boltzmann path re-derives after mutating F_res, so the final `J_form` is consistent. Brittle if a future contributor adds a residual term after the second derivative without re-deriving. | `forms_logc_muh.py:839` + `boltzmann.py:386` | Opus-B | Maintainability hazard; not active bug. |
| **F11** | **Note** | `phi_clamp = 50.0` is validated only as `> 0`. With z=±2, `phi_clamp > ~350` would overflow `exp(700)` in fp64. Current production safe; no upper-bound guard. | Boltzmann phi clamp | Opus-B | Recommend `phi_clamp ≤ 150` guard. |
| **F12** | **Note** | No Picard oscillation detection in `picard_outer_loop_general` — only max-iter exit. Strong Stern coupling could induce two-cycle oscillation that ω=0.5 damping cannot quench; would hit max-iter and trigger the linear-phi IC fallback without a specific diagnostic. | `picard_ic.py` | Sonnet-B | Falls back cleanly but no oscillation-specific signal. |
| **F13** | **Note** | Nested `adj.stop_annotating()` (demo outer + `solve_anchor_with_continuation` inner + internal Stage-3 wrap) relies on pyadjoint's refcount-style nesting being correct. Between Stage 2 exit and Stage 3 entry, annotation is ON, but no `fd.solve`/`fd.assemble` runs in that window — safe today, but no module-level guard. | demo + solver wraps | Sonnet-A, Opus-A | Implicit assumption; document it. |
| **F14** | **Note** | `set_stern_capacitance_model(ctx, 0)` is silently accepted. C_S = 0 ≠ no-Stern Dirichlet limit; it produces an insulator (zero-charge) Robin BC rather than the proper no-Stern path. Demo's no-Stern uses C_S = 100 F/m² (effectively a hard ramp toward Dirichlet). | `anchor_continuation.py:set_stern_capacitance_model` | Opus-A | Document or assert. |

---

## What's verified clean (worth being explicit)

These are the high-risk items the audit specifically traced and confirmed correct:

| Concern | Status | Evidence |
|---------|--------|----------|
| **Cross-stage `ctx['_last_solver']` aliasing** (Stage 3 reusing Stage 1's solver/forms) | **CLEAN** | `_build_for_voltage` (`grid_per_voltage.py:1010-1036`) builds fresh ctx/F_res/J_form/NonlinearVariationalProblem/Solver per voltage. Stage 2's solver is unreachable after `solve_anchor_with_continuation` returns. The `ctx['_last_solver']` slot is assigned at line 1032 but never read internally — solver is passed by parameter to `warm_walk_phi`. (Opus-C) |
| **`warm_walk_phi._march` recursive state restoration** | **CLEAN** | ckpt_outer/ckpt_inner pattern at `grid_per_voltage.py:271-303` restores U via `_restore_U` AND explicitly re-asserts `paf.assign(v_prev_substep)` because `_restore_U` doesn't touch paf. All success/failure paths within the recursion terminate consistently. (Opus-C). Final-SS failure path is separate and benign here (F3). |
| **`U_prev.assign(U)` placement in SER** | **CLEAN** | At `grid_per_voltage.py:192`, immediately after `solver.solve()` and before observable assemble + dt update. SER pseudo-time integration is intact. (Opus-C) |
| **`AdaptiveLadder` `prev=0` infinite-loop risk** | **CLEAN** | `__init__` rejects non-positive `initial_scales`. Geometric branch only runs when `previous_scale > 0`; warm-start-floor branch uses arithmetic midpoint. `_inserts_at_current_step >= max_inserts_per_step (=4)` raises `LadderExhausted`. (Opus-C, Sonnet-C) |
| **`snapshot_U` / `restore_U` atomicity** | **CLEAN** | Snapshot saves both U and U_prev as deep copies; restore puts both back. SER's "previous solution" is correctly preserved across rollbacks. (Opus-C, Sonnet-C) |
| **Stern unit conversion in `set_stern_capacitance_model`** | **CLEAN** | Forward and reverse helpers symmetric; `conv_factor = V/(F·c·L)` at `forms_logc_muh.py:248-254` is unit-consistent. (Opus-C) |
| **Boltzmann closed-form invariant (Cs⁺/SO₄²⁻ NOT in Newton state)** | **CLEAN** | No `TrialFunction`/`TestFunction`/`Function on W` leakage in `build_steric_boltzmann_expressions`. Outputs are UFL expressions in φ only. (Opus-B) |
| **Picard singular-Jacobian guard** | **CLEAN** | Explicit `|det| < 1e-300` and non-finite guards at `picard_ic.py:748-757` and `1418-1432`, with clean failure path to fallback. (Opus-B) |
| **Stern bisection bracket failure** | **GUARDED** | Explicit `f_lo * f_hi > 0` check at `picard_ic.py:259` → closed-form linear-Debye fallback (no silent meaningless midpoint). See F2 for Picard-IC-quality nuance. (Opus-B) |
| **UFL min/max smoothness at clip plateau** | **CLEAN** | Code uses `fd.min_value/max_value` (UFL smooth), not Python `min/max`. Jacobian goes to zero on the BV-residual contribution at the plateau, but full Newton Jacobian remains well-conditioned because BV is one of many residual contributions. Production V band is unclipped at `exponent_clip=100`. (Opus-B) |
| **μ_H NP flux reduction** | **CLEAN** | Form correctly reduces to `D·c·(∇u_H + em·z_H·∇φ)` when the μ_H substitution is undone. (Opus-B) |
| **Stern Robin sign convention** | **CLEAN** | `F_res -= C_S·(φ_applied − φ)·w ds` — sign matches cathodic-polarization convention. (Opus-B) |
| **Bikerman shared-θ closure self-consistency** | **CLEAN** | Total volume occupied by Cs⁺ + SO₄²⁻ + dynamic species is bounded by 1; closure is mathematically self-consistent (proof in Opus-B report). |
| **Observable sign chain** | **CLEAN** | `R_j > 0` (cathodic) × `n_e/2 > 0` × `−I_SCALE` (demo) yields negative current density at cathodic overpotentials, then `_to_json_list` × demo convention produces positive `cd_mA_cm2` for cathodic — consistent with deck. (Opus-C) |
| **`with_solver_options` immutability** | **CLEAN** | Returns a new frozen `SolverParams` via `dataclasses.replace`. `sp_anchor` and `sp_baseline` are independent. (Sonnet-A, Opus-A) |
| **Cross-factor state isolation** | **CLEAN** | Mesh built once, but shared is fine (Firedrake mesh coordinates immutable). `SolverParams` frozen. `_make_sp` returns new instances. Each `solve_anchor_with_continuation` builds fresh ctx with anonymous Constants. No cross-factor leakage. (Opus-A, Sonnet-A) |
| **Numerical mesh grading sum** | **CLEAN** | For Ny=80, β=3.0, D=1.0: `y[i] = (i/80)³` gives `y[80] = 1.0` FP-exact, `ΣΔy = 1.0` FP-exact. Electrode at y=0 (graded fine end), marker 3 matches Firedrake convention referenced by observables. (Opus-A) |
| **Physical a_nondim derivation** | **CLEAN** | For H⁺ r=2.8 Å: `a_nondim = 6.645e-5`; `1/a_phys = 1.806e4 mol/m³`. Matches CLAUDE.md Hard Rule #7 to 2 sig figs. (Opus-A) |
| **`_factor_label` uniqueness** | **CLEAN** | All four factors (1.0, 1e-6, 1e-12, 1e-18) produce distinct labels. (Sonnet-A, Opus-A) |
| **`per_point_callback` only on success** | **CLEAN** | `grid_per_voltage.py:1112` gates the callback on `ok=True`. `_grab` never sees a diverged ctx. (Opus-A, Sonnet-A) |
| **Stage 2 failure → Stage 3 unreachable with corrupt U** | **CLEAN** | Unconditional early-return at demo:464 before Stage 3 is reachable; `snes_error_if_not_converged=True` ensures divergence raises. (Sonnet-A, Opus-A) |

---

## Agreement / disagreement

**High agreement:**
- All 6 agents agree no critical correctness bugs in the codepath.
- Both Chunk-C agents converge on the cross-stage solver aliasing being safe (fresh ctx per voltage).
- Both Chunk-B agents converge on the free_dyn floor being an algebraically-correct cap with latent numerical-only risk.
- Both Chunk-A agents converge on cross-factor state isolation, mesh grading, and `_factor_label` correctness.

**One disagreement (F2 — Stern bisection fallback):**
- **Sonnet-B (MEDIUM):** The linear-Debye fallback when bracket fails "does NOT satisfy the Robin identity — it is an approximation seeded into the Picard IC. The downstream full solver enforces the Stern BC anyway, so convergence absorbs this, but the Picard IC can be measurably wrong in this path. No warning is emitted."
- **Opus-B (CLEAN):** "Explicit `f_lo * f_hi > 0` check at line 259 with linear-Debye closed-form fallback at line 268. No silent meaningless midpoints."
- **Resolution:** Both agents observed the same code. They differ on framing: is "approximate IC seed when bracket fails" a correctness defect or graceful degradation? For correctness purposes — the downstream solver does enforce the Robin BC, so the *final answer* is correct. The IC quality is suboptimal in this rare branch but doesn't cause wrong results. **Graded as F2 medium / downstream-corrected** — track but don't block.

**One semi-disagreement (F3 — `warm_walk_phi` final-SS failure):**
- **Sonnet-C (MEDIUM):** Flagged that the final `run_ss` outside the bisection has no `_restore_U` on failure.
- **Opus-C (CLEAN):** Cleared the recursive bisection paths but didn't specifically audit the final-SS-failure path.
- **Resolution:** Sonnet's finding is a real footgun, but **benign in this demo** because Stage 3 builds fresh ctx per voltage (so corrupt ctx from a failed voltage doesn't leak to the next). Worth fixing if anyone refactors to share ctx across voltages.

---

## What this audit did NOT cover

- **Firedrake/PETSc internals.** We trust the SNES, KSP, and form-assembly machinery.
- **The cation_hydrolysis path** (`_run_gamma_picard`). Sonnet-C flagged a "MEDIUM: silently accepts Newton-diverged U" finding there, but this path requires `cation_hydrolysis_enabled=True` which this demo does not set. Worth investigating if you ever enable that path.
- **The `c_s_ladder` parameter** to `solve_anchor_with_continuation`. The demo uses `_last_solver.solve()` manually instead. (Sonnet-C's pass-1 mis-flag of Stage 2 traced to this code path; the demo doesn't use it.)
- **Numerical convergence behavior on edge cases** (e.g., V_RHE = −0.5 V with extreme k0 ratios). The audit verifies the code is structurally correct; whether Newton actually converges at every grid point is empirical.
- **Adjoint correctness** (since this demo only does forward). Adjoint tape hygiene is flagged in F13 as an implicit assumption.

---

## Final verdict

**PASS — no critical correctness bugs in the codepath as exercised by this demo.**

The implementation is unusually defensive for research code: explicit bracket guards, singular-Jacobian checks, immutable `SolverParams`, fresh per-voltage ctx, error-on-non-convergence SNES. The 14 findings above are latent risks for future configurations or refactors, not active bugs.

The two items that come closest to "would matter in this demo if anything unusual happened":
1. **F1** — if a future change weakens `make_bv_solver_params` and `sp[10]` is incomplete, PETSc would silently pick wrong defaults and the anchor build would silently fail to converge. An assertion at solver creation would be cheap insurance.
2. **F6** — `_to_json_list` masking `inf` as `null` on converged points: if the BV rate ever overflows at a converged point, the JSON output would say "missing" rather than "blew up". Currently safe because production V band is well-bounded.

Per-chunk reports preserved at `.verification/correctness/*.md` (sonnet/opus × A/B/C).
