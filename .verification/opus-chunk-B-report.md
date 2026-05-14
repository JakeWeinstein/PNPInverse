# Opus Chunk B Verification Report

**Scope:** `Forward/bv_solver/forms_logc_muh.py`, `Forward/bv_solver/boltzmann.py`, `Forward/bv_solver/picard_ic.py`

**Doc audited:** `docs/solver/forward_codepath_demo_slide15.md`

---

## Findings

### CRITICAL-1 — Misattribution: Newton solver build is NOT in `build_forms_logc_muh`
- **SEVERITY:** critical
- **LOCATION:** doc lines 56–58 (Stage 1 step 2); contradicts
  `Forward/bv_solver/forms_logc_muh.py:166`–`882` and
  `Forward/bv_solver/anchor_continuation.py:1101`–`1107`
- **DESCRIPTION:** The doc lists, inside `build_forms_logc_muh`, the steps
  `NonlinearVariationalProblem(F, U, bcs, J)`,
  `NonlinearVariationalSolver(problem, options=SNES_OPTS_CHARGED)`, and
  `→ ctx['_last_solver']  ← reused by Stern bump`.
  The actual `build_forms_logc_muh` stops at building `F_res`, `J_form`,
  and `bcs`, then `ctx.update(...)` and `return ctx` (lines 839–882).
  It never constructs a `NonlinearVariationalProblem` or
  `NonlinearVariationalSolver`, and never writes `ctx['_last_solver']`.
- **EVIDENCE:**
  - `forms_logc_muh.py:839` — `J_form = fd.derivative(F_res, U)`
  - `forms_logc_muh.py:841`–`876` — `ctx.update({...})` (no solver keys)
  - `forms_logc_muh.py:882` — `return ctx`
  - `grep` confirms no `NonlinearVariationalProblem` / `NonlinearVariationalSolver` /
    `SNES_OPTS_CHARGED` / `_last_solver` anywhere in `forms_logc_muh.py`.
  - Actual construction is in `anchor_continuation.py:1101`–`1107`
    (and `grid_per_voltage.py:430`–`447`, `1026`–`1032`).

### CRITICAL-2 — `SNES_OPTS_CHARGED` does not exist
- **SEVERITY:** critical
- **LOCATION:** doc line 57 — `options=SNES_OPTS_CHARGED`
- **DESCRIPTION:** No symbol `SNES_OPTS_CHARGED` (or `SNES_OPTS`) exists
  anywhere in `Forward/`. The actual options used in
  `anchor_continuation.py:1099`–`1105` are computed at call time from
  the user-supplied `params['solver_options']` dict, filtered against
  `NON_PETSC_KEYS`, with one default added
  (`solve_opts.setdefault("snes_error_if_not_converged", True)`).
- **EVIDENCE:**
  - `grep -rn "SNES_OPTS" Forward/` → no hits.
  - `anchor_continuation.py:1099` — `solve_opts = {k: v for k, v in items if k not in NON_PETSC_KEYS}`
  - `anchor_continuation.py:1104`–`1106` — `solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)`

### WARNING-1 — "2×2 scalar Picard on (ψ_S, ψ_D)" is a mathematical misattribution
- **SEVERITY:** warning
- **LOCATION:** doc line 63 (Stage 1 step 3 sub-bullet under
  `picard_outer_loop_general`)
- **DESCRIPTION:** The doc states the Picard loop is a "2×2 scalar
  Picard on (ψ_S, ψ_D)". Inspection of `picard_outer_loop_general`
  (`picard_ic.py:1254`–`1606`) shows the iterated state is `R = [R_1,
  …, R_N]` (per-reaction surface rates, N reactions), not (ψ_S, ψ_D).
  - The Picard primary variable is the rate vector `R`. Each iteration
    builds an N×N linear system via `_assemble_n_reaction_system`
    (`picard_ic.py:1016`–`1088`) — for the parallel 2e/4e stack N=2 so
    the resulting linear solve is 2×2 (`_solve_2x2`,
    `picard_ic.py:1091`–`1105`), but the unknowns are `(R_0, R_1)`.
  - (ψ_S, ψ_D) are byproducts updated each iter by
    `_solve_picard_stern_split` (`picard_ic.py:385`–`462`) — a 1-D
    bisection on the Stern Robin closure (`solve_stern_split` at
    `picard_ic.py:181`–`296`) for the single-ion case, or a closed-form
    linear-Debye match for the multi-ion case. Neither is a 2×2 system.
- **EVIDENCE:**
  - `picard_ic.py:1342` — `R = [0.0] * N`  (Picard primary vector)
  - `picard_ic.py:1376`–`1445` — iteration body relaxes `R`
  - `picard_ic.py:1416` — `R_solve, det = _solve_linear_system(M_mat, b_vec)`
  - `picard_ic.py:1091` `_solve_2x2(M, b) -> (R0, R1, det)` — confirms
    the 2×2 lives in rate-space, not ψ-space.
- **RECOMMENDED REWRITE:** "N-reaction Picard outer loop on per-reaction
  surface rates `R = (R_1, …, R_N)`. Stern split `(ψ_S, ψ_D)` recomputed
  each iter via 1-D Robin bisection (single-ion) or linear-Debye match
  (multi-ion)."

### WARNING-2 — Picard fallback condition is broader than "oscillates"
- **SEVERITY:** warning
- **LOCATION:** doc line 65 — "falls back to linear_phi if Picard oscillates"
- **DESCRIPTION:** The fallback at
  `forms_logc_muh.py:1036`–`1039` triggers on any `not ok` return from
  `_try_debye_boltzmann_ic_muh`. That includes:
  - `picard_max_iters_delta=…` (non-convergence / oscillation),
  - `singular_jacobian_iter_k_det=…`,
  - `non_finite_R_iter_k`, `non_finite_state_iter_k`,
  - `n_species_lt_3`, `empty_reactions`, `mu_h_idx_unsupported`,
  - `no_boltzmann_counterion`, `multi_ion_phi_o_solve_failed: …`,
  - `species_arity_mismatch`, `h_idx_out_of_range`.
  "Oscillates" describes only the max-iters failure mode. The actual
  policy is "fall back on any Picard failure".
- **EVIDENCE:**
  - `forms_logc_muh.py:1036`–`1039` —
    ```
    ctx["initializer_fallback"] = True
    ctx["initializer_fallback_reason"] = reason
    ctx["initializer_picard_iters"] = picard_iters
    set_initial_conditions_logc_muh(ctx, solver_params)
    ```
  - `picard_ic.py:1329`, `1331`, `1335`, `1339`, `1423`–`1442`, `1466`–`1476`,
    `1519`–`1532` enumerate the failure-reason strings the muh body
    forwards.

### WARNING-3 — Function-name inconsistency (`set_ic_debye_boltzmann_logc_muh`)
- **SEVERITY:** warning
- **LOCATION:** doc line 61 (Stage 1 step 3), and "Files touched" panel line 208
- **DESCRIPTION:** Doc uses the shorthand `set_ic_debye_boltzmann_logc_muh`.
  The actual function is
  `set_initial_conditions_debye_boltzmann_logc_muh` at
  `forms_logc_muh.py:997`. Line number itself is correct, but a reader
  grepping for the doc name will miss the real function.
- **EVIDENCE:**
  - `forms_logc_muh.py:997` — `def set_initial_conditions_debye_boltzmann_logc_muh(...)`
  - `dispatch.py:34`,`37`,`116`,`136` reference the full name.

### WARNING-4 — `build_model_scaling` lives in `Nondim/transform.py`, not `nondim.py`
- **SEVERITY:** warning (note for non-Firedrake readers; could mislead a
  grepper into thinking it's in `Forward/bv_solver/nondim.py`)
- **LOCATION:** doc line 48 — "`build_model_scaling            nondim.py`"
- **DESCRIPTION:** `build_model_scaling` is defined in
  `Nondim/transform.py:172` (top-level `Nondim` package). The
  `Forward/bv_solver/nondim.py` is a different module that hosts
  `_add_bv_scaling_to_transform` / `_add_bv_reactions_scaling_to_transform`,
  which `build_forms_logc_muh` does call subsequently. The doc collapses
  the two distinct files into one label.
- **EVIDENCE:**
  - `forms_logc_muh.py:61` — `from Nondim.transform import build_model_scaling, _get_nondim_cfg, _bool`
  - `forms_logc_muh.py:70` — `from .nondim import _add_bv_scaling_to_transform, _add_bv_reactions_scaling_to_transform`
  - `forms_logc_muh.py:206` — `base_scaling = build_model_scaling(...)`

### NOTE-1 — `add_boltzmann_counterion_residual(skip_bikerman=True)` is a no-op for the demo's bikerman stack
- **SEVERITY:** note
- **LOCATION:** doc line 53 — implies `add_boltzmann_counterion_residual`
  contributes to the residual for Cs⁺/SO₄²⁻
- **DESCRIPTION:** For the Cs⁺/SO₄²⁻ demo both counterions have
  `steric_mode="bikerman"`; `build_forms_logc_muh` calls
  `add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)`
  at line 881. With `skip_bikerman=True`, the inner loop skips every
  bikerman entry (line 365), so for this demo the call only walks the
  list and returns the entry count without modifying `F_res`. The real
  Poisson source for Cs⁺ and SO₄²⁻ is added inside `build_forms_logc_muh`
  itself (line 658: `F_res -= z_scale_shared * charge_rhs *
  charge_density_total * w * dx`).
- **EVIDENCE:**
  - `forms_logc_muh.py:879`–`881` — call with `skip_bikerman=True`
  - `boltzmann.py:365` — `if skip_bikerman and entry.get("steric_mode") == "bikerman": continue`
  - `forms_logc_muh.py:652`–`660` — the real bikerman charge density gets
    added to Poisson inside `build_forms_logc_muh`
- **IMPACT:** Doc reader may believe the bikerman counterion Poisson
  source is added by `add_boltzmann_counterion_residual`. In the Cs⁺/SO₄²⁻
  demo it is NOT — that helper handles only ideal entries on this call.

### NOTE-2 — `W = V_u × V_μH × V_φ` notation glosses over the actual mixed-space construction
- **SEVERITY:** note
- **LOCATION:** doc line 42 — `mixed FE space  W = V_u × V_μH × V_φ`
- **DESCRIPTION:** `build_context_logc_muh` (line 152) simply wraps
  `build_context_logc` (`forms_logc.py:59`), which builds
  `V_scalar = fd.FunctionSpace(mesh, "CG", order)` and
  `W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])`.
  All `n_species + 1` blocks are identical CG-`order` scalar spaces;
  the muh transform changes only the *interpretation* of the H⁺ block
  (it's now μ_H, not u_H), not its function space. The doc's
  `V_u × V_μH × V_φ` triple is mathematically suggestive but slightly
  misleading because every block uses the same scalar space.
- **EVIDENCE:**
  - `forms_logc_muh.py:152`–`163` — `build_context_logc_muh` body
    delegates to `build_context_logc` and sets
    `ctx["logc_muh_transform"] = True`
  - `forms_logc.py:105`–`106` — actual mixed-space construction
- **IMPACT:** Harmless for understanding the math but a reader expecting
  literally three differently-typed FE spaces will be confused. The
  module docstring at `forms_logc_muh.py:152`–`160` already notes "muh
  transform changes only the *interpretation* of `U.sub(mu_h_idx)`, not
  the function space."

---

## Verified Claims (PASS list)

| Doc claim | Status | Evidence |
|---|---|---|
| `forms_logc_muh.py:152` is `build_context_logc_muh` | PASS | line 152: `def build_context_logc_muh(...)` |
| `forms_logc_muh.py:166` is `build_forms_logc_muh` | PASS | line 166: `def build_forms_logc_muh(...)` |
| `forms_logc_muh.py:997` is `set_initial_conditions_debye_boltzmann_logc_muh` | PASS | line 997 (name shortened in doc, see WARNING-3) |
| `boltzmann.py:91` is `build_steric_boltzmann_expressions` | PASS | line 91 exact |
| `boltzmann.py:272` is `add_boltzmann_counterion_residual` | PASS | line 272 exact |
| `build_forms_logc_muh` calls `build_model_scaling` | PASS | `forms_logc_muh.py:206` |
| `build_forms_logc_muh` calls `_resolve_mu_h_index` | PASS | `forms_logc_muh.py:290` |
| `build_forms_logc_muh` calls `build_steric_boltzmann_expressions` with `(1-A_dyn)` for Cs⁺ AND SO₄²⁻ | PASS | `forms_logc_muh.py:418`, `boltzmann.py:204`,`206`,`254` — both bundles share `free_dyn = (1 - A_dyn_local)` |
| `build_forms_logc_muh` calls `add_boltzmann_counterion_residual` | PASS (with caveat NOTE-1) | `forms_logc_muh.py:881` (`skip_bikerman=True`) |
| `J_form = derivative(F_res, U)` inside `build_forms_logc_muh` | PASS | `forms_logc_muh.py:839` |
| `set_initial_conditions_debye_boltzmann_logc_muh` calls `picard_outer_loop_general` | PASS | `forms_logc_muh.py:1070`,`1267` |
| Composite-ψ + multispecies-γ seed (Layer 2) | PASS | bikerman branch: `forms_logc_muh.py:1462`–`1500`; multi-ion: `1382`–`1416` |
| Falls back to linear-phi IC | PASS (broader trigger than "oscillates"; see WARNING-2) | `forms_logc_muh.py:1036`–`1039` |
| `_build_eta_clipped` clamps `eta_scaled` BEFORE `α·n_e` | PASS — matches Hard Rule #2 | `forms_logc_muh.py:388`–`403` (clip on `eta_scaled = bv_exp_scale * eta_raw`); used at lines 555, 586 inside `±α·n_e * eta_j` |
| `exponent_clip = 100` is the production default | PASS | `config.py:138`,`163`; doc Hard Rule #2 consistent |
| Tresset Eq. (19) closure with (1−A_dyn) numerator | PASS — code formula matches Tresset on the full-equilibrium subset (A_dyn≡0), and the `(1-A_dyn)` factor is the hybrid extension (documented in `writeups/May13th/analytic_counterion_derivation.tex:344`–`350`) | `boltzmann.py:204`,`241`,`254` |
| Bikerman closure is shared-θ across all bikerman entries | PASS | `boltzmann.py:241`–`242` (single `denom` summed over `per_ion_q`); each bundle reuses `denom` and `free_dyn` (line 254) |
| Cs⁺ and SO₄²⁻ never enter Newton state vector (Layer 4) | PASS — `build_steric_boltzmann_expressions` returns UFL expressions for `c_steric`, used in the Poisson source (`forms_logc_muh.py:652`–`660`) and the dynamic-species packing closure (`forms_logc_muh.py:446`–`452`); no `U.sub(...)` slot is allocated for them | confirmed |
| Convergence Layer 2 (debye_boltzmann IC seeds composite ψ_S+ψ_D so Newton initial residual is O(1)) | PASS in spirit. The "O(1) vs O(1e26)" is a qualitative argument; in the saturated regime `exp(-z·φ_applied)` with `φ_applied/V_T ≈ 21` for V=0.55 V already pushes the bare residual past 1e9, and saturation factors with clamped exponent 100 can in principle produce `exp(100) ~ 2.7e43`. 1e26 is a defensible rough estimate, not a tight bound. | qualitative |

---

## Cross-Chunk Notes

- The Stern bump using `set_stern_capacitance_model` + `ctx['_last_solver'].solve()` (doc Stage 2) is implemented in
  `anchor_continuation.py:412`–`460` and `1264`–`1280` — owned by Chunk C.
- The doc's Stage 1 `_last_solver` attribution to `build_forms_logc_muh`
  (CRITICAL-1) shifts the actual ownership of `_last_solver` creation
  to `anchor_continuation.py:1107` — that file is Chunk C scope.
- The `formulation="logc_muh"` wiring, `THREE_SPECIES_LOGC_BOLTZMANN`
  catalog, and `exponent_clip=100` knob plumbing are Chunk A.

---

## VERDICT: ISSUES FOUND

**Summary:**
- 2 CRITICAL issues — both in the "Stage 1 step 2 / `build_forms_logc_muh`"
  bullet:
  1. Newton-solver construction is misattributed to `build_forms_logc_muh`
     when it actually happens in `anchor_continuation.py:1101`–`1107`.
  2. The cited `SNES_OPTS_CHARGED` symbol does not exist in the codebase.
- 4 WARNING-level issues:
  - "2×2 scalar Picard on (ψ_S, ψ_D)" mis-identifies the Picard primary
    variable (actually `R = (R_1, …, R_N)` rate vector).
  - Picard fallback fires on broader conditions than "oscillates".
  - Function-name shorthand (`set_ic_debye_boltzmann_logc_muh`) does not
    match the actual name (`set_initial_conditions_debye_boltzmann_logc_muh`)
    even though the line number is right.
  - `build_model_scaling` is in `Nondim/transform.py`, not the local
    `Forward/bv_solver/nondim.py` the doc implies.
- 2 NOTE-level items (no-op `skip_bikerman=True` call, slightly imprecise
  mixed-space notation).

Most line/file annotations check out. The substantive misattributions
center on the solver-construction ownership (which actually crosses
into Chunk C) and the Picard variable identity. The convergence-layer
descriptions are accurate, including the precise clip location
(Hard Rule #2) and the Bikerman closure structure.
