# Verification Report — Chunk C
## Scope: anchor_continuation.py, grid_per_voltage.py, observables.py

This report verifies the documentation claims in
`docs/solver/forward_codepath_demo_slide15.md` against the actual
implementation in the three files in Chunk C.

---

## Summary of findings

- **9 issues found** total: 2 critical (incorrect symbol names),
  6 warnings (stale/imprecise fields, wrong formula transcription),
  1 note (imprecise math description).
- The overall structure of Stage 1, Stage 2, and Stage 3 is **accurate
  in spirit**: line numbers point at real functions, the convergence
  mechanisms (k0 ladder, sqrt-midpoint insertion, Stern bump,
  preconverged-anchor handoff, warm walk with bisection, plateau
  detection) all exist and behave as the doc describes at a high level.
- The doc has multiple **stale field names** in `PreconvergedAnchor`
  (`phi_eta`/`dof_count` vs `phi_applied_eta`/`mesh_dof_count`) and
  one **wrong function name** in two places
  (`solve_reaction_k0_model` vs `set_reaction_k0_model`).
- The `set_stern_capacitance_model` ctx key is wrong
  (`stern_capacitance_func` vs the real `stern_coeff_const`).
- The `make_run_ss` plateau formula in the doc is **not literally the
  code**: doc shows additive `|Δ| < rel·|j| + abs`, code uses
  `max`-denominator + OR (`(Δ/sv <= rel) or (Δ <= abs)`).

---

## Detailed findings

### CRITICAL #1 — Wrong function name `solve_reaction_k0_model`

- **SEVERITY**: critical
- **LOCATION**: doc lines 73 and the table at line 179 (Layer 1 description)
- **DESCRIPTION**: The doc's Stage 1 pseudocode and Layer-1 row both
  refer to `solve_reaction_k0_model` at `anchor_continuation.py:246`.
  No such function exists.  The actual function at line 246 is
  `set_reaction_k0_model`. The "solve_" prefix is misleading — the
  function only writes a value into the metadata dict and the live FE
  `Function`; it does NOT solve anything.
- **EVIDENCE**:
  - Doc line 73: `solve_reaction_k0_model(ctx, j, k0_scale * k_target)
    anchor_continuation.py:246`.
  - `anchor_continuation.py:246`:
    `def set_reaction_k0_model(ctx: dict, j: int, k0_model_value: float) -> None:`
  - `anchor_continuation.py:1194`:
    `for j, k_target in k0_targets.items(): set_reaction_k0_model(...)`

### CRITICAL #2 — Wrong ctx key `stern_capacitance_func`

- **SEVERITY**: critical
- **LOCATION**: doc line 104 (Stage 2 Stern bump description)
- **DESCRIPTION**: The doc says
  `set_stern_capacitance_model` "mutates `ctx['stern_capacitance_func']`
  ← shared FE Constant". The actual ctx key is `stern_coeff_const`
  (a Firedrake `Constant`, not a `Function`).  The mechanism described
  is correct in spirit — a shared FE Constant baked into the residual
  is mutated in place via `.assign()` so the residual sees the new
  C_S without rebuild — but the literal ctx key name is wrong.
- **EVIDENCE**:
  - Doc line 104: `• mutates ctx['stern_capacitance_func']  ← shared FE Constant`
  - `anchor_continuation.py:448`:
    `stern_const = ctx.get("stern_coeff_const")`
  - `anchor_continuation.py:467`:
    `stern_const.assign(nondim_value)`
  - `forms_logc_muh.py:667`:
    `stern_coeff = fd.Constant(float(stern_capacitance_model))`
  - `forms_logc_muh.py:863`:
    `"stern_coeff_const": stern_coeff,`

### WARNING #1 — `PreconvergedAnchor` field name `phi_eta` is wrong

- **SEVERITY**: warning
- **LOCATION**: doc line 108 (Stage 2 → PreconvergedAnchor emission)
- **DESCRIPTION**: The doc says
  `PreconvergedAnchor(phi_eta, U_snapshot, k0_targets, dof_count, ladder_history)`.
  Actual field name is `phi_applied_eta`, NOT `phi_eta`.
- **EVIDENCE**:
  - `anchor_continuation.py:133`: `phi_applied_eta: float`
  - Demo `solver_demo_slide15_no_speculative_cs.py:504`:
    `phi_applied_eta=float(anchor_v_rhe) / V_T,`

### WARNING #2 — `PreconvergedAnchor` field name `dof_count` is wrong

- **SEVERITY**: warning
- **LOCATION**: doc line 109
- **DESCRIPTION**: The doc lists `dof_count` as a field; actual name
  is `mesh_dof_count`.
- **EVIDENCE**:
  - `anchor_continuation.py:136`: `mesh_dof_count: int`
  - Demo line 510: `mesh_dof_count=mesh_dof_count,`

### WARNING #3 — `make_run_ss` plateau formula is not literally the code

- **SEVERITY**: warning
- **LOCATION**: doc lines 80-83 (Stage 1 SS-plateau pseudocode) and
  line 184 (Layer 6 description)
- **DESCRIPTION**: The doc shows the plateau predicate as
  `|Δj_cd| < ss_rel_tol · |j_cd| + ss_abs_tol`. The code uses an
  `OR` of two normalized tests with a `max`-denominator:
  ```python
  delta = abs(fv - prev_flux)
  sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
  is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)
  ```
  In particular the doc's "+ ss_abs_tol" is additive; the code's
  `ss_abs_tol` only enters as (a) a floor for the relative-test
  denominator, and (b) a separate absolute-test branch.  These are
  not algebraically equivalent.  At small fluxes the code is **more
  forgiving** than the doc's formula suggests (the OR-branch triggers
  via the absolute test alone when |Δ| < abs_tol regardless of |j_cd|).
- **EVIDENCE**:
  - Doc line 81-82: `if |Δj_cd| < ss_rel_tol · |j_cd| + ss_abs_tol for ss_consec steps:`
  - `grid_per_voltage.py:195-197`: see code excerpt above.

### WARNING #4 — Doc claim "consecutive counter resets on failure"

- **SEVERITY**: warning
- **LOCATION**: doc / OPUS-DEEPER item I ("consecutive counter resets on failure")
- **DESCRIPTION**: The doc / task prompt asserts the consecutive
  plateau counter resets on failure.  The code resets `steady_count`
  to 0 only when the plateau test is False (line 198,
  `steady_count = steady_count + 1 if is_steady else 0`).  Newton
  failure causes an early `return False` from `run_ss` (line 191) —
  there is no counter to reset because the closure exits.  The "reset"
  is on "non-steady-detection", NOT on "Newton failure". Semantically
  similar but the task's framing is imprecise.
- **EVIDENCE**:
  - `grid_per_voltage.py:188-191`: `try: solver.solve(); except Exception: return False`
  - `grid_per_voltage.py:198`: `steady_count = steady_count + 1 if is_steady else 0`

### WARNING #5 — Stage 1 pseudocode uses `for j, k_target in k0_targets:` (not `.items()`)

- **SEVERITY**: warning
- **LOCATION**: doc line 72
- **DESCRIPTION**: The pseudocode `for j, k_target in k0_targets:`
  would iterate the dict's **keys** in Python (and then unpacking
  would fail because keys are ints, not pairs).  The real code uses
  `k0_targets.items()`.  This is a minor doc-pseudocode issue but a
  literal reading is wrong.
- **EVIDENCE**:
  - `anchor_continuation.py:1083, 1091, 1194, 1390, 1548`: all use
    `for j, k_target in k0_targets.items():`

### WARNING #6 — `warm_walk_phi` kwarg name is `bisect_depth` (not `bisect_depth_warm`)

- **SEVERITY**: warning
- **LOCATION**: doc line 139 (Stage 3 grid walk pseudocode)
- **DESCRIPTION**: Doc shows
  `warm_walk_phi(..., n_substeps=8, bisect_depth_warm=5)`. The actual
  signature uses `bisect_depth` (default 3); the orchestrator
  `solve_grid_with_anchor` takes a parameter named `bisect_depth_warm`
  (default 5) and passes it through as
  `bisect_depth=bisect_depth_warm`. This is a transcription error.
- **EVIDENCE**:
  - `grid_per_voltage.py:218-226`:
    `def warm_walk_phi(..., n_substeps: int = 4, bisect_depth: int = 3, ...)`
  - `grid_per_voltage.py:1095-1096`: `n_substeps=n_substeps_warm, bisect_depth=bisect_depth_warm`

### NOTE #1 — "32× refinement" math description

- **SEVERITY**: note
- **LOCATION**: doc line 186 (Layer 8)
- **DESCRIPTION**: Doc says "up to `bisect_depth_warm=5` levels (32×
  refinement)".  Strictly, the refinement at depth `k` is `2^k = 32`
  applied to the **failed substep interval**, but each recursive call
  re-uses `n_substeps=8` over the halved interval, so the **finest
  substep** is `8 · 32 = 256×` smaller than the depth-0 substep size.
  The doc's "32× refinement" is correct interpretation-wise (each
  level halves the interval being marched), but a reader could
  mistakenly read it as "the finest substep is 32× smaller". Minor.
- **EVIDENCE**:
  - `grid_per_voltage.py:271-303` (`_march` recursion structure).

---

## Verification of remaining claims (PASS items)

### 1. Function locations (line numbers)

All `file:line` annotations point to real functions:

| Doc claim | Actual location | Status |
|---|---|---|
| `anchor_continuation.py:902` = `solve_anchor_with_continuation` | line 902 def | PASS |
| `anchor_continuation.py:719` = `AdaptiveLadder` | line 719 class | PASS |
| `anchor_continuation.py:246` symbol name | actually `set_reaction_k0_model` | FAIL (Critical #1) |
| `grid_per_voltage.py:875` = `solve_grid_with_anchor` | line 875 def | PASS |
| `grid_per_voltage.py:218` = `warm_walk_phi` | line 218 def | PASS |
| `grid_per_voltage.py:135` = `make_run_ss` | line 135 def | PASS |
| `observables.py:67` = `_build_bv_observable_form` | line 67 def | PASS |

### 2. `solve_anchor_with_continuation` defaults

- `initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0)` — VERIFIED at
  `anchor_continuation.py:907`.
- `max_inserts_per_step=4` — VERIFIED at line 908.
- `ic_at_target=True` — VERIFIED at line 916.

### 3. `AdaptiveLadder` API

- `is_done()` — VERIFIED at line 825.
- `current_scale` property — VERIFIED at line 811-816.
- `record_success()` — VERIFIED at line 829-836 (resets
  `_inserts_at_current_step = 0`).
- `record_failure_and_insert()` — VERIFIED at line 838-887.
- Sqrt-midpoint insertion — VERIFIED at line 884:
  `midpoint = math.sqrt(prev * scale)` (geometric mean = correct
  for a multiplicative ladder).  The doc's "sqrt(prev_scale * failed_scale)"
  is correct.

### 4. OPUS item A — sqrt vs geometric mean

The literal code is `math.sqrt(prev * scale)`. `prev` is
`previous_scale = self._planned[self._idx - 1]` (last
**successful** scale).  `scale` is `self._planned[self._idx]`
(the rung that **just failed**).  So the midpoint is the geometric
mean of (last success, failed) — exactly the doc's "sqrt-midpoint" /
"geometric mean".  CORRECT.

### 5. OPUS item B — `max_inserts_per_step=4` semantics

- `_inserts_at_current_step` is the counter; reset to 0 in
  `record_success`.
- Cap check is `>= self._max_inserts_per_step` (line 862-863).
- **Per-rung**, not per-ladder cumulative.  Doc's "per-step cap on
  ladder midpoint inserts" wording is consistent.

### 6. OPUS item C — `ic_at_target=True` branch

- When TRUE (default): IC is built at production k0 (no pre-ramp).
- When FALSE: k0 is ramped to floor BEFORE `set_initial_conditions`
  so Picard sees tiny rates.
- Branching at `anchor_continuation.py:1079-1085` matches the doc's
  description.

### 7. OPUS item D — `k0_targets` data structure

- `k0_targets` is `Dict[int, float]` per the signature at line 906.
- All loops use `.items()`. The doc's pseudocode `for j, k_target in
  k0_targets:` is shorthand for `.items()`.

### 8. OPUS item E — `ctx['_last_solver'].solve()` reuse during Stern bump

- The reuse is mechanically correct because the Stern coefficient is
  a `fd.Constant` baked into the residual (`forms_logc_muh.py:667-668`):
  ```python
  stern_coeff = fd.Constant(float(stern_capacitance_model))
  F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)
  ```
- Mutating it via `stern_coeff.assign(new_val)` updates the form for
  the next `solver.solve()` without rebuilding.
- However the doc names the ctx key `stern_capacitance_func` (wrong)
  vs actual `stern_coeff_const`.  See Critical #2.
- The reuse pattern itself is confirmed by the demo at
  `solver_demo_slide15_no_speculative_cs.py:454-459`.

### 9. OPUS item F — `PreconvergedAnchor` fields

Actual fields (`anchor_continuation.py:111-137`):
```
phi_applied_eta: float      # doc says "phi_eta"          → WRONG
U_snapshot: tuple           # doc says "U_snapshot"       → CORRECT
k0_targets: tuple           # doc says "k0_targets"       → CORRECT
mesh_dof_count: int         # doc says "dof_count"        → WRONG
ladder_history: tuple       # doc says "ladder_history"   → CORRECT
```

See Warning #1 and Warning #2.

### 10. OPUS item G — `solve_grid_with_anchor` sorting

- One-time static sort at line 1039-1044 by `|phi - anchor.phi_applied_eta|`.
- The **source-pool** (line 1048-1050) starts with the anchor and
  grows via `sources.append((target_phi, snap))` on each success.
- Source choice inside the loop is `min(sources, key=|s[0] - target_phi|)`
  — nearest to the **current target**, not nearest to the anchor.
- Anchor IS the initial source; doc claim verified.

### 11. OPUS item H — `_march` recursive bisection semantics

- `v_prev_substep` is the **last SUCCEEDED v_sub** (initialized to
  `v0`, only updated on success at line 285).
- On failure, the bisection midpoint is `0.5 * (v_prev_substep +
  v_sub)` — between the last-succeeded substep and the failing one.
- The recursion order is
  `_march(v_prev_substep, v_mid, ...)` then
  `_march(v_mid, v_sub, ...)` — left-then-right.
- Source-state restoration uses `ckpt_outer` (snapshot at function
  entry) on full bail-out and `ckpt_inner` (per-substep snapshot)
  for the immediate-failed-substep rollback.
- The dedicated comment at line 272-277 explains why `paf` is NOT a
  reliable v_prev — `_restore_U` doesn't roll `paf` back.  Good
  defensive code.

### 12. OPUS item I — `make_run_ss` plateau

- Uses `ss_rel_tol`, `ss_abs_tol`, `ss_consec` as param names (verified
  at line 143-145).
- Consecutive counter resets to 0 when `is_steady` is False (line
  198).
- Newton failure exits the closure via `return False` — no counter
  reset needed.
- See Warning #3 for the formula transcription issue and Warning #4
  for the "reset on failure" framing.

### 13. OPUS item J — observable `(n_e/2)·R_j` interpretation

- `N_ELECTRONS_REF = 2` is a module constant (observables.py:43).
- The literal weight is `(n_e_j / N_ELECTRONS_REF)`, where the "2"
  is the reference electron count that anchors `I_SCALE`
  (cross-referenced to `scripts/_bv_common.compute_i_scale`).
- The "/2" is **NOT** a collection efficiency (Mangan/Ruggiero
  N=0.224 would have been a separate factor applied externally, not
  baked into the observable form).
- The doc's "(n_e/2)·R_j" is shorthand for the actual `(n_e_j /
  N_ELECTRONS_REF)·R_j`.  Mathematically correct.

### 14. OPUS item K — `bisect_depth_warm=5 → 32× refinement`

- 2^5 = 32 — correct for "how many times the marched interval is
  halved".
- See Note #1 for the subtlety about substep size (the doc's
  refinement-factor wording could mislead, but the math is fine).

### 15. OPUS item L — `solve_grid_with_anchor` is the Phase 5γ/6 successor

- CLAUDE.md hard rule #1 says C+D is
  `solve_grid_per_voltage_cold_with_warm_fallback`.
- The doc here uses `solve_grid_with_anchor` (Phase 5γ MVP per its
  own docstring at line 894-959; "Use
  `solve_anchor_with_continuation` upstream to build the anchor").
- The two functions coexist in `grid_per_voltage.py` (line 314 and
  line 875 respectively).
- CLAUDE.md confirms: "For Phase 6α/β work prefer the newer
  `solve_anchor_with_continuation` + `solve_grid_with_anchor` pair".
- Confirmed.

### 16. Layer 7 — Newton's basin radius (~0.05 V) claim

- This is a heuristic, not a code claim, so cannot be code-verified.
  The 25-point grid over `[-0.4, +0.55]` V is `0.95/25 ≈ 0.04 V`
  spacing, consistent with the doc's "0.05 V basin" remark.

### 17. SS plateau detection observable

- The observable is built with `mode="current_density"` (line 1108-1110
  of anchor_continuation.py, line 1033-1035 of grid_per_voltage.py).
- `_build_bv_observable_form` returns an electron-weighted sum of
  reaction rates over `ds(electrode_marker)` (observables.py:106-121).
- Doc's "Σ_j (n_e/2)·R_j ds" is correct in form.

---

## File:line annotations table (consolidated)

| Doc reference | Verified | Notes |
|---|---|---|
| `anchor_continuation.py:902` solve_anchor_with_continuation | YES | matches def |
| `anchor_continuation.py:719` AdaptiveLadder | YES | matches class |
| `anchor_continuation.py:246` "solve_reaction_k0_model" | NO  | actual is `set_reaction_k0_model` |
| `grid_per_voltage.py:875` solve_grid_with_anchor | YES | matches def |
| `grid_per_voltage.py:218` warm_walk_phi | YES | matches def |
| `grid_per_voltage.py:135` make_run_ss | YES | matches def |
| `observables.py:67` _build_bv_observable_form | YES | matches def |

---

## VERDICT: ISSUES FOUND

The doc is structurally accurate and useful as a roadmap.  The
critical issues are **typos / stale field-names / wrong ctx keys**
that would lead a reader writing follow-on code to use the wrong
identifier (`solve_reaction_k0_model` does not exist; the
PreconvergedAnchor field is `phi_applied_eta` not `phi_eta`; the ctx
key is `stern_coeff_const` not `stern_capacitance_func`).

Mathematical claims about convergence mechanisms are correct in
intent.  The make_run_ss formula and the "consec counter resets on
failure" framing have minor imprecisions worth fixing for technical
accuracy.

Suggested edits (in priority order):

1. Line 73 and line 179: replace `solve_reaction_k0_model` with
   `set_reaction_k0_model`.
2. Line 104: replace `ctx['stern_capacitance_func']` with
   `ctx['stern_coeff_const']`.
3. Lines 108-109: replace `phi_eta` with `phi_applied_eta`, replace
   `dof_count` with `mesh_dof_count`.
4. Line 139: replace `bisect_depth_warm=5` with `bisect_depth=5`
   in the `warm_walk_phi(...)` call signature pseudocode (the kwarg
   passed by solve_grid_with_anchor is `bisect_depth=bisect_depth_warm`).
5. Lines 80-82 and line 184: clarify the `make_run_ss` plateau
   formula matches the actual OR-of-two-tests
   `(Δ/max(|j|,|j_prev|,abs_tol) <= rel_tol) or (Δ <= abs_tol)`.
6. Line 72: change `for j, k_target in k0_targets:` to
   `for j, k_target in k0_targets.items():` to be literal-Python.
