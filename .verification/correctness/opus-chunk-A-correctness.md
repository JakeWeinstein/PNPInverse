# Opus correctness audit — Chunk A (solver demo + dispatch + mesh + nondim + _bv_common)

Scope: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`,
`Forward/bv_solver/dispatch.py`, `Forward/bv_solver/mesh.py`,
`Forward/bv_solver/nondim.py`, and the relevant pieces of
`scripts/_bv_common.py` (`make_bv_solver_params`,
`setup_firedrake_env`, `SolverParams.with_solver_options`).

## Q1 — Stage 2 bump-failure path (corrupt-U risk before Stage 3)

- SEVERITY: note (no bug found; intentional guard works)
- LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:454-498`
- TRIGGER + EVIDENCE: On any rung exception, `bump_err` is set,
  `bump_history` records the failure, the `for` loop exits via
  `break` at line 463, then the `if bump_err is not None:` block at
  line 464 unconditionally returns a "stern-bump-failed" report.
  Stage 3 cannot be reached on the failure path — verified
  textually that lines 502+ are guarded by the early `return` at
  464-498. Additionally, `_last_solver` is constructed inside
  `solve_anchor_with_continuation` with `snes_error_if_not_converged`
  defaulting to True (`anchor_continuation.py:1100`), so a diverged
  Newton raises rather than silently leaving a partial-Newton U.
  Result: Stage 3 only ever sees a fully-converged U. **No
  corrupt-U regression risk on the bump-failure path.**

## Q2 — Cross-iteration state for K0_R4E_FACTORS

- SEVERITY: note (no bug — isolation is by-construction)
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:740-746`
  (per-factor loop), `_make_sp` 126-244
- TRIGGER + EVIDENCE:
  1. `mesh` is built once outside the loop but is NEVER mutated
     after construction (only coords stretched at build time in
     `mesh.py:107`); both `solve_anchor_with_continuation` and
     `solve_grid_with_anchor` build their own ctx atop the shared
     mesh, so DOF-map state is per-ctx.
  2. `_make_sp` returns a brand-new `SolverParams` per call (frozen
     `@dataclass`; `dataclasses.replace`-based mutators). Two calls
     for the anchor and baseline Stern values inside one factor
     iteration each construct their own `bv_bc`/`bv_convergence`
     dicts via `make_bv_solver_params` (`_bv_common.py:1063`).
  3. Each `solve_anchor_with_continuation` call builds a fresh ctx
     with new `fd.Constant` objects (e.g. `stern_coeff` at
     `forms_logc_muh.py:667`); these Constants are never registered
     in a globally named registry — they live only on the new ctx.
  Cross-factor Constant-name collision risk: **none** (Constants
  created without names use anonymous UFL labels; ctx and form trees
  are fresh per iteration). No persistence of factor N's mutated
  Constants into factor N+1.

## Q3 — Physical a_nondim derivation

- SEVERITY: note (formula correct; numbers match CLAUDE.md Hard
  Rule 7)
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:107-118`
- TRIGGER + EVIDENCE: Demo uses
  `a_phys = (4/3) π r³ N_A`  (m³/mol), then
  `a_nondim = a_phys · C_SCALE`. The demo hard-codes
  `_C_SCALE = 1.2`, matching `_bv_common.C_SCALE = C_O2 = 1.2`
  (`_bv_common.py:86, 133`). For r = 2.80 Å (H₃O⁺ Stokes),
  computed values:

  - `a_phys(H⁺) = 5.537e-5 m³/mol`
  - `a_nondim(H⁺) = 6.645e-5`
  - Dimensional Bikerman c_max = `1/a_phys = 1.806e4 mol/m³`

  This matches the CLAUDE.md Hard Rule 7 prediction (≈1.8 × 10⁴
  mol/m³) and is ≈150× higher than the `A_DEFAULT=0.01` clamp
  (120 mol/m³). Convention is consistent with `A_OH_HAT` derivation
  in `_bv_common.py:200-202` (same `(4/3)πr³N_A · C_SCALE` formula).
  **No unit/scale bug.**

## Q4 — with_solver_options mutation semantics

- SEVERITY: note (returns new immutable instance; sp_anchor and
  sp_baseline cannot share state)
- LOCATION: `Forward/params.py:178-180`
- TRIGGER + EVIDENCE:
  ```
  def with_solver_options(self, opts):
      return dataclasses.replace(self, solver_options=opts)
  ```
  `SolverParams` is `@dataclass(frozen=True)` (line 29). `replace`
  builds a fresh frozen instance. The original is untouched.
  Demo line 238-242 builds `new_opts = dict(sp.solver_options)` and
  `new_bv = dict(new_opts["bv_convergence"])` — both shallow copies
  before mutation; then `sp.with_solver_options(new_opts)` returns
  a new sp. **Stage-1 anchor sp and Stage-3 baseline sp are
  independent.** No cross-state risk.

## Q5 — `_stern_bump_ladder` edge cases

- SEVERITY: warning (one degenerate-but-survivable case;
  one undefended)
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:304-322`
- TRIGGER + EVIDENCE: Manually traced for each requested input:

  | target  | returned                              | notes |
  |---------|---------------------------------------|-------|
  | 0.10    | `[0.10]`                              | redundant single solve at anchor C_S — no-op SNES |
  | 0.20    | `[0.20]`                              | direct 0.10 → 0.20 (2× growth, OK) |
  | 0.30    | `[0.20, 0.30]`                        | intermediate at 0.20 then final 0.30 |
  | 100.0   | `[0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0]` | the verified ladder, target lands exactly on last rung — appended once (good) |
  | 0.05    | `[0.05]`                              | DOWN-bump (target < STERN_ANCHOR) — function silently accepts and does a single 0.10→0.05 step |

  - **target = 0.10**: returns `[0.10]`; Stage 2 still calls
    `set_stern_capacitance_model(ctx, 0.10)` and `solver.solve()` —
    Newton starts converged so this should be a 0-iter solve. Wastes
    one residual evaluation but is correct.
  - **target < STERN_ANCHOR (e.g. 0.05)**: docstring says
    "bump targets" and `STERN_ANCHOR -> target via ... 5× growth per
    rung". Function silently treats this as a single-step path
    (`[target]`). For a 2× shrink this is fine numerically, but it
    violates the documented "up to target" contract. The demo never
    exercises this from the CLI (default 0.20 / 100.0), so this is
    a latent **WARNING** for future callers, not a current bug.

## Q6 — dispatch.py default branching

- SEVERITY: warning (silent default to "logc" for any unrecognized
  formulation string; documented as defensive only, but exposes a
  config typo as a silent backend swap)
- LOCATION: `Forward/bv_solver/dispatch.py:68-79`
- TRIGGER + EVIDENCE:
  ```
  def _resolve_backend(solver_params):
      formulation = _read_formulation(solver_params)
      if formulation == "logc_muh":
          return "logc_muh"
      return "logc"
  ```
  - Only `"logc_muh"` triggers the muh backend. ANY other string
    (`"logc"`, `"LOGC"`, `"conc"`, `"foo"`, empty string, missing
    config) falls through to `logc`. `_read_bv_convergence_field`
    already calls `.strip().lower()` (line 57), so case sensitivity
    is normalized — but `formulation="logmuh"` (typo missing `c_`)
    silently falls back to `logc` instead of erroring.
  - The docstring acknowledges this is defensive because
    `config.py:_validate_formulation` rejects unknown names at
    parse time. **If that upstream validator is ever bypassed
    (e.g. test fixtures that construct SolverParams directly),
    typos will route silently.** Severity: **warning** (latent).
  - Initializer dispatch (line 113-120) has the same shape: anything
    other than `"debye_boltzmann"` falls through to `linear_phi`.
    Same silent-default behavior, same warning class.

## Q7 — Mesh grading arithmetic for Nx=8, Ny=80, β=3.0, D=1.0

- SEVERITY: note (FP-exact result; no off-by-one or roundoff bug)
- LOCATION: `Forward/bv_solver/mesh.py:68-108`
- TRIGGER + EVIDENCE: Node positions are
  `y_k = (k/Ny)^β · D` for `k ∈ [0, Ny]`. With Ny=80, β=3.0, D=1.0:

  - `y[0] = 0.0`
  - `y[80] = (80/80)³ · 1.0 = 1.0` **exactly** (FP-exact:
    `1.0³·1.0 = 1.0`).
  - `Σ Δy = y[80] − y[0] = 1.0` exactly (telescoping).
  - First cell width: `Δy₀ = (1/80)³ ≈ 1.953e-6`.
  - Last cell width: `Δy₇₉ ≈ 3.70e-2`.
  - Min Δy at index 0 (electrode side); max Δy at index 79 (bulk
    side). **Fine end is at y=0**, which matches the
    `RectangleMesh` convention where marker 3 = bottom (y=0). The
    docstring (`mesh.py:84-86`) labels marker 3 as electrode — this
    matches the demo's expectation (electrode at y=0, graded
    fine).

  No FP roundoff bug. The unit-cube `coords[:, 1] **= β` step is
  numerically exact at the boundary because `(0.0)**3 = 0.0` and
  `(1.0)**3 = 1.0`. **For non-unit `domain_height_hat`**, the
  multiply-by-D step might introduce 1-ULP drift at the top
  boundary, but the demo always uses
  `domain_height_hat = L_EFF_M / 1e-4 = 1.0`, so this isn't
  exercised here.

## Q8 — pyadjoint tape lifecycle

- SEVERITY: warning (tape is RECORDING between Stage 2 exit and
  Stage 3 entry; would taint an adjoint-context caller)
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:397-543`
- TRIGGER + EVIDENCE:
  - Stage 1 wraps `solve_anchor_with_continuation` in
    `with adj.stop_annotating():` (lines 397-404).
  - Stage 2 wraps `_last_solver.solve()` per rung in
    `with adj.stop_annotating():` (lines 457-458).
  - Between Stage 2 exit and Stage 3 entry (lines 502-514), code
    runs **with annotation ON** but only:
    - `snapshot_U(ctx_anchor["U"])` — pure `data_ro.copy()` ndarray
      ops (`grid_per_voltage.py:102-103`); no Firedrake `solve`
      calls and no tape ops.
    - `PreconvergedAnchor(...)` construction — frozen dataclass; no
      Firedrake ops.
  - Stage 3's `solve_grid_with_anchor` internally wraps its main
    loop in `with adj.stop_annotating():`
    (`grid_per_voltage.py:1060`).
  - `_grab` callback (lines 520-538) calls
    `fd.assemble(_build_bv_observable_form(...))` — `assemble` of
    a UFL form pushes onto the tape if annotation is on. But this
    runs inside `solve_grid_with_anchor`'s `stop_annotating()`
    block, so it's safe.

  **Net result for the demo as a script (no adjoint context):**
  annotation status doesn't matter — there's no `tape.compute_gradient`
  consumer downstream. **For a hypothetical adjoint-driver that
  calls this function**: the construction window between lines
  502-514 is safe (no tape ops); but the demo also calls
  `with_phi_applied`, `dict(...)`, and `_make_sp` from inside the
  caller's annotation context. **No actual tape pollution** because
  none of these call `fd.solve` / `fd.assemble`. Severity is
  **warning** only because the demo doesn't *guarantee* annotation
  hygiene at module scope — a future maintainer who adds an
  `fd.assemble(...)` in the construction window would silently
  taint the tape.

## Q9 — `_grab` failure paths

- SEVERITY: note (callback never sees failed-solve ctx)
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:520-538` and
  `Forward/bv_solver/grid_per_voltage.py:609-614, 681-687,
  1107-1113`
- TRIGGER + EVIDENCE: Every call site for `per_point_callback`
  inside `grid_per_voltage.py` is gated on the local `snap is not
  None` / `ok` flag and is invoked **only on success**:

  - Line 609-614 (cold phase): `if snap is not None: ... if
    per_point_callback is not None: per_point_callback(...)`.
  - Line 1107-1113 (anchor-walk): `if ok: ... if
    per_point_callback is not None: per_point_callback(...)`.

  Failed grid points never trigger the callback, so `_grab` never
  sees a diverged ctx. The internal `try/except` in `_grab` is
  defensive only (catches the case where `_build_bv_observable_form`
  fails on the converged state — e.g. degenerate stoichiometry).
  **No assemble-on-partial-state bug.**

## Q10 — Other findings

### Q10.1 — `_to_json_list` silently coerces `inf` to None on converged points

- SEVERITY: warning (latent — masks solver overflows as None)
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:560-564`
- EVIDENCE: `[float(x) if (np.isfinite(x) and converged[i]) else
  None ...]`. If a converged point assembles an `inf` current
  density (e.g. exponent_clip saturation producing overflow in
  observable form), `isfinite(inf) == False` → value silently
  recorded as `None`. The companion `converged[]` array would still
  say `True`, creating an inconsistent record. Recommend logging or
  preserving an explicit "inf" sentinel separately from the
  "not-converged" None.

### Q10.2 — `set_stern_capacitance_model` accepts `c_s_f_m2 = 0`

- SEVERITY: note (documented; not a bug, but a silent no-op risk)
- LOCATION: `Forward/bv_solver/anchor_continuation.py:444-447`
- EVIDENCE: Only `< 0` raises. `c_s_f_m2 == 0` is allowed and
  silently sets the Stern coefficient to zero — which doesn't
  match the no-Stern Dirichlet limit (that's a build-time
  decision). With C_S=0, the Robin BC becomes
  `eps·∂φ/∂n = 0` (insulator), NOT the no-Stern
  `φ_s = φ_m` Dirichlet. Caller responsibility, not a demo bug
  (demo never passes 0), but worth flagging.

### Q10.3 — Demo per-rung try/except over-broad

- SEVERITY: note
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:455-463`
- EVIDENCE: `except Exception as exc:` catches everything,
  including `KeyboardInterrupt`-adjacent
  `SystemExit`/`BaseException` cases would still propagate (good),
  but catches errors like `KeyError` from a mis-spelled
  `ctx_anchor["_last_solver"]` that should crash loudly. Mostly
  cosmetic — the failure mode is recorded in `bump_err` and the
  driver returns cleanly. Recommend `except (RuntimeError,
  fd.exceptions.ConvergenceError)` or similar to surface
  programmer errors.

### Q10.4 — `n_substeps_warm`, `bisect_depth_warm` shadowed at module scope but unused for cold-only paths

- SEVERITY: note
- LOCATION: `solver_demo_slide15_no_speculative_cs.py:89-90, 549`
- EVIDENCE: Passed through to `solve_grid_with_anchor`; consumed
  correctly. No bug; flagging for completeness.

### Q10.5 — Constant-name collision risk (sanity check)

- SEVERITY: note (verified clean)
- EVIDENCE: `forms_logc_muh.py:667` creates
  `fd.Constant(float(stern_capacitance_model))` without an explicit
  `name=` kwarg. Firedrake/UFL Constants are identity-based, not
  name-based, so two factor iterations each create a unique
  `Constant` instance on their own ctx. No accidental sharing.

## VERDICT

No critical correctness bugs found in the audited scope.

- **Q1 (Stage 2 corruption)**: clean — early-return guarantee
  preserves the invariant.
- **Q2 (cross-factor state)**: clean — fresh `SolverParams` + fresh
  ctx per factor.
- **Q3 (physical a_nondim)**: clean — formula and numbers match
  CLAUDE.md Hard Rule 7 (c_max(H⁺) = 1.806 × 10⁴ mol/m³).
- **Q4 (`with_solver_options`)**: clean — immutable via
  `dataclasses.replace`.
- **Q5 (`_stern_bump_ladder`)**: clean for the demo's exercised
  inputs (0.20, 100.0). Warning: down-bumps (target <
  STERN_ANCHOR) silently collapse to single-step, violating
  docstring "up to target" contract.
- **Q6 (dispatch default)**: warning — silent fallback to `"logc"`
  on any unrecognized formulation/initializer string; defensive
  only, but a latent typo trap.
- **Q7 (mesh grading)**: clean — FP-exact: y[80] = 1.0, ΣΔy = 1.0
  exactly; electrode at y=0 with marker 3 matches docstring.
- **Q8 (tape lifecycle)**: clean for the demo (no fd.solve /
  fd.assemble in the annotation-on construction window). Warning:
  no module-level guarantee against future maintainers adding a
  tape-tainting op there.
- **Q9 (`_grab` failures)**: clean — callback only fires on
  converged points.
- **Q10**: minor latent issues (inf→None coercion, broad
  `Exception` catch, `c_s=0` silent no-op). None block the demo.

Bottom line: the demo is correctness-safe as written. The warnings
above are latent and would only fire on inputs the demo doesn't
currently send (down-bumps, formulation typos, adjoint-context
callers). Severity is "warning, not critical" across the board.
