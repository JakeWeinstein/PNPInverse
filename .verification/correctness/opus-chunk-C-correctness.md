# Opus Chunk C — Code Correctness Audit

Targeted audit of:
- `Forward/bv_solver/anchor_continuation.py` — PreconvergedAnchor, AdaptiveLadder,
  set_reaction_k0_model, set_stern_capacitance_model, solve_anchor_with_continuation
- `Forward/bv_solver/grid_per_voltage.py` — make_run_ss, warm_walk_phi, solve_grid_with_anchor
- `Forward/bv_solver/observables.py` — full

---

## Q1 — AdaptiveLadder midpoint pathology

**SEVERITY: LOW (mitigations are in place; one minor edge case worth noting).**

Code: `anchor_continuation.py:838-887`.

- **prev=0 case → infinite loop:** *Not possible at construction.*
  `__init__` (`anchor_continuation.py:773-777`) raises `ValueError` if `any(s <= 0.0
  for s in initial_scales)`. So `previous_scale` returned by the property
  (`anchor_continuation.py:819-823`) can only ever be either `None` (idx==0) or a
  strictly-positive entry. The geometric branch `math.sqrt(prev * scale)` cannot see
  `prev=0`. The first-rung path (prev is None) does NOT call `math.sqrt`; it uses the
  arithmetic-bisection branch only when `warm_start_floor is not None`, and that branch
  has an explicit guard `not (warm_start_floor < midpoint < scale)` →
  `return False` (line 879-880). **No infinite-loop risk.**

- **prev≈scale FP collapse (sqrt → prev):** Possible in principle, but the inserted
  midpoint is appended at position `self._idx` (line 881/885), shifting the failing
  rung to `_idx+1`. On the very next iteration `current_scale` reads
  `self._planned[self._idx]` which is the **new** midpoint. If `midpoint == prev` by
  FP equality, the orchestrator just re-attempts at essentially the same scale, and on
  failure inserts yet another midpoint — bounded by `max_inserts_per_step=4`, after
  which `record_failure_and_insert` returns `False` (line 862-863). The orchestrator
  then raises `LadderExhausted` (e.g. `anchor_continuation.py:1492-1494`). **No
  unbounded loop, but no special-case detection of the collapse; ladder simply
  exhausts.** Acceptable.

- **`max_inserts_per_step` exhausted:** `record_failure_and_insert` returns `False`
  (line 862-863). Caller checks return; in `_run_k0_ladder` this yields the
  `(False, k0_lad, k0_last_ok_snap)` tuple (line 1226) and the top-level orchestrator
  converts to `LadderExhausted` (e.g. line 1492-1494). **Behaves as documented.**

LOCATION: `anchor_continuation.py:838-887`
TRIGGER: Repeated failures at the same rung.
EVIDENCE: All paths reach a bounded `return False` → orchestrator raises
`LadderExhausted`. No infinite loop possible.

---

## Q2 — AdaptiveLadder.is_done off-by-one

**SEVERITY: NONE (correct).**

Code: `anchor_continuation.py:825-827`:

```
def is_done(self) -> bool:
    return self._idx >= len(self._planned)
```

`record_success` advances `_idx += 1` (line 835). After the last rung (scale=1.0)
succeeds at idx=`len(planned)-1`, post-increment makes `_idx == len(planned)`, and
`is_done` returns True via `>=`. The while-loop guard `while not k0_lad.is_done()`
(line 1192) terminates correctly. **No off-by-one.**

LOCATION: `anchor_continuation.py:825-827, 829-836`
EVIDENCE: `>=` comparison is correct; loop exits on the next `is_done` check.

---

## Q3 — `sp[10]` indexing tuple-order risk

**SEVERITY: MEDIUM (lurking refactor hazard — not a current bug).**

Code: `anchor_continuation.py:1095`:

```
params_block = sp[10] if hasattr(sp, "__getitem__") else {}
```

And `grid_per_voltage.py:984`:

```
n_s, order, dt, t_end, z_v, D_v, a_v, _, c0, phi0, params = solver_params
```

Both consistently treat slot 10 as the PETSc params dict. The 11-tuple destructuring
in `solve_grid_with_anchor` (line 984) is order-validated: any field reordering
would change `params` to be the wrong field and the next caller would see a wrong
dict. However the call site in `anchor_continuation.py:1095` reads via `sp[10]`
which has **no semantic checking** — pure positional. If the SolverParams shape
changes upstream, this site silently reads the wrong block until something else
breaks (PETSc error on a bogus key, or runs with default opts). **No current bug,
but brittle to refactor.** A `SolverParams.params` accessor would be safer; current
code is just positional.

LOCATION: `anchor_continuation.py:1095`; cross-ref `grid_per_voltage.py:984`.
TRIGGER: SolverParams field reorder/insertion before slot 10.
EVIDENCE: No assertion that sp[10] is a dict at this site; only the destructure form
performs structural validation.

---

## Q4 — `ctx['_last_solver']` reuse semantics across Stage 2 → Stage 3 (PRIORITY A)

**SEVERITY: NONE (correctly handled — each stage builds a fresh solver).**

I traced `solve_grid_with_anchor` end-to-end. The key path:

1. Outer `for visit_n, orig_idx in enumerate(visit_order):` loop
   (`grid_per_voltage.py:1061`).
2. Each iteration calls `_build_for_voltage(target_phi)`
   (`grid_per_voltage.py:1010-1036`), which:
   - Builds a **NEW** `ctx` via `build_context(sp_v, mesh=mesh)` (line 1012).
   - Builds **NEW** forms via `build_forms(ctx, sp_v)` (line 1013).
   - Constructs a **NEW** `NonlinearVariationalProblem` with `ctx["F_res"]` etc.
     (line 1026-1028).
   - Constructs a **NEW** `NonlinearVariationalSolver` (line 1029-1031).
   - Assigns `ctx["_last_solver"] = solver` (line 1032).
3. That fresh `solver` is passed into `warm_walk_phi(..., solver=solver, ...)`
   (line 1089-1105).

So Stage 2's `ctx['_last_solver']` is **discarded** the moment
`solve_grid_with_anchor` builds a new ctx. The Stage 2 bump-loop solver (built at
`anchor_continuation.py:1101-1107`) holds a problem reference to Stage 2's `F_res`
and `U` — those tie to Stage 2's ctx, which goes out of scope after
`solve_anchor_with_continuation` returns. Stage 3's solver is constructed against
Stage 3's `F_res` / `U` / `bcs`. **No cross-stage form-leak.**

Notable subtle point: `ctx['_last_solver']` is set inside `_build_for_voltage` but
is then never read inside `solve_grid_with_anchor` itself — the `solver` is passed
by parameter. The dict slot exists for downstream code (diagnostics or callbacks).
Even if a callback read `ctx['_last_solver']`, it would be the Stage 3 solver
matching Stage 3's ctx, because the assignment happens AFTER the new solver is
built. **Correct.**

LOCATION: `grid_per_voltage.py:1010-1036` (`_build_for_voltage`).
EVIDENCE: Fresh `build_context` + `build_forms` + `NonlinearVariationalProblem` +
`NonlinearVariationalSolver` per voltage iteration.

---

## Q5 — set_stern_capacitance_model unit conversion

**SEVERITY: NONE (correct conversion, with proper validation).**

Code: `anchor_continuation.py:444-467`. The conversion is:

```
factor = float(scaling.get("bv_stern_phys_to_nondim_factor", 1.0))
nondim_value = float(c_s_f_m2) * factor
stern_const.assign(nondim_value)
```

`factor` is computed at form-build time in `forms_logc_muh.py:248-254`:

```
conv_factor = potential_scale_v / (_F * concentration_scale * length_scale)
scaling["bv_stern_phys_to_nondim_factor"] = float(conv_factor)
```

Units check: physical C_S [F/m²] = C/(V·m²). `conv_factor` has units
`V / (C/mol · mol/m³ · m) = V·m² / C`. Their product is dimensionless, as required
for `bv_stern_capacitance_model` (which multiplies a dimensionless residual). The
`else` branch (line 256-257) sets `factor=1.0` when `nondim` is disabled or stern
is None — but that branch is also paired with the `stern_const is None` guard at
`anchor_continuation.py:448-454` which raises before any use. **Both branches
internally consistent.** No off-by-power-of-10.

Reverse direction (`get_stern_capacitance_model`, line 470-490) is the inverse:
`float(stern_const) / factor`. **Symmetric and unit-consistent.**

LOCATION: `anchor_continuation.py:444-467`, `forms_logc_muh.py:238-257`.

---

## Q6 — snapshot_U / restore_U pair

**SEVERITY: NONE (semantics correct, with caveat about `paf` documented).**

Code: `grid_per_voltage.py:102-109`:

```
def _snapshot_U(U) -> tuple:
    return tuple(d.data_ro.copy() for d in U.dat)

def _restore_U(snap: tuple, U, U_prev) -> None:
    for src, dst in zip(snap, U.dat):
        dst.data[:] = src
    U_prev.assign(U)
```

`snapshot_U` saves **U only**, not U_prev.
`restore_U` writes U from the snapshot and then re-assigns `U_prev := U` (the
restored U). After restore, U == U_prev, i.e. the system is at "no transient
difference" — exactly the state at the beginning of a new SS run (where
`U_prev.assign(U)` happens implicitly via the next solve loop, or rather, the SER
state `(U - U_prev)/dt` is zero). **This is correct for re-entering the SER loop
fresh.** Whatever U_prev was BEFORE restore is lost, but for SER restart from a
checkpoint that's the desired behavior.

LOCATION: `grid_per_voltage.py:102-109`.

---

## Q7 — warm_walk_phi `_march` state restoration (PRIORITY B)

**SEVERITY: NONE (state-management is correct; deserves the explicit comment
inside the source).**

Code: `grid_per_voltage.py:271-303`. Detailed trace:

```
def _march(v0, v1, depth):
    substeps = np.linspace(v0, v1, n_substeps + 1)[1:]
    ckpt_outer = _snapshot_U(U)            # state at v0 (entry)
    v_prev_substep = v0
    for v_sub in substeps:
        ckpt_inner = _snapshot_U(U)        # state at v_prev_substep
        paf.assign(v_sub)
        if run_ss(...): success           # advance: v_prev_substep = v_sub
            v_prev_substep = v_sub
            continue
        # FAILURE PATH
        _restore_U(ckpt_inner, U, U_prev)  # U back to v_prev_substep state
        paf.assign(v_prev_substep)         # paf back to v_prev_substep
        if depth >= bisect_depth:
            _restore_U(ckpt_outer, U, U_prev)  # back to v0 state
            paf.assign(v0)
            return False
        v_mid = 0.5 * (v_prev_substep + v_sub)
        if not _march(v_prev_substep, v_mid, depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(v0)
            return False
        if not _march(v_mid, v_sub, depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(v0)
            return False
        v_prev_substep = v_sub
    return True
```

Verification:

- After substep failure: state is **restored to ckpt_inner = state at
  v_prev_substep**, and `paf.assign(v_prev_substep)`. So recursion
  `_march(v_prev_substep, v_mid, ...)` starts at the valid state corresponding to
  its first argument. **Correct.**
- If bisect_depth exceeded: state restored to ckpt_outer (state at v0) and
  `paf.assign(v0)`. Returns False. Caller sees fully-rolled-back state — no
  corruption. **Correct.**
- If first inner recursion succeeds but the second fails: state restored to
  ckpt_outer + paf at v0 before returning False. So even partial progress is
  rolled back; caller sees clean v0 state. **Correct.**
- Note: the explicit `paf.assign(v_prev_substep)` after `_restore_U(ckpt_inner,...)`
  is required because `_restore_U` does not touch `paf` (paf is not in U.dat;
  the in-source comment lines 272-278 documents this rationale). The follow-up
  recursive calls then drive paf forward from `v_prev_substep`.

LOCATION: `grid_per_voltage.py:271-303`.
EVIDENCE: All failure paths restore both U and paf consistently; all success paths
advance v_prev_substep before next iteration. State invariants hold.

---

## Q8 — paf rollback at grid-point boundary

**SEVERITY: NONE (caller assigns paf at the start of each point).**

In `solve_grid_with_anchor`, each grid-point iteration:

1. Calls `_build_for_voltage(target_phi)` which builds a **fresh ctx** (line 1064).
   The new ctx has its OWN `phi_applied_func` — Stage 2's paf is gone.
2. Calls `ctx["phi_applied_func"].assign(float(src_phi))` (line 1087) to set paf
   to the warm-walk source voltage BEFORE calling warm_walk_phi.
3. warm_walk_phi internally advances paf via substeps. On _march failure, paf is
   restored to v_anchor_eta (the function-entry value, which equals src_phi).

So even if warm_walk_phi failed, paf would be back at src_phi — and the next grid
iteration discards the entire ctx and builds a new one with a fresh paf. **No
cross-iteration paf leak.**

LOCATION: `grid_per_voltage.py:1064-1087`.

---

## Q9 — make_run_ss plateau false-positive on sign-flip through zero

**SEVERITY: LOW (theoretically possible, ss_consec=4 mitigates).**

Code: `grid_per_voltage.py:193-211`. Steady-detection:

```
delta = abs(fv - prev_flux)
sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)
steady_count = steady_count + 1 if is_steady else 0
```

Sign-flip scenario: fv passes from +eps to -eps. `delta = |fv - prev_flux| = 2*eps`,
`sv = max(eps, eps, ss_abs_tol)`. For small eps near ss_abs_tol, `sv = ss_abs_tol`,
so `delta/sv = 2*eps/ss_abs_tol` — if 2*eps <= ss_abs_tol, it's flagged steady. But
this only counts ONE plateau hit; `ss_consec=4` consecutive plateau hits are
required (line 211). After the sign flip, the next iteration's |delta| is back to
"actual" physics-driven step size, resetting `steady_count`. **Bounded.** A pure
zero-crossing at near-equilibrium with tiny step size is the only way to trigger a
false positive — physically unlikely in BV-PNP runs where fv is far from zero in
the production grid. **Acceptable.**

LOCATION: `grid_per_voltage.py:193-211`.

---

## Q10 — make_run_ss dt update on "ok"

**SEVERITY: LOW (correct: dt grows on shrinking delta, not on raw Newton convergence).**

Code: `grid_per_voltage.py:199-207`:

```
if prev_delta is not None and delta > 0:
    ratio = prev_delta / delta
    if ratio > 1.0:                      # delta shrinking
        grow = min(ratio, dt_growth_cap)
        dt_val = min(dt_val * grow, dt_max)
    else:                                # delta growing/equal
        dt_val = max(dt_val * 0.5, float(dt_init))
    dt_const.assign(dt_val)
```

The "ok" you reference is the Newton solve success at line 188-190 (caught by
try/except → `return False` on failure). dt update is gated on `prev_delta is not
None and delta > 0`, i.e. on **the observable's delta history**, not on Newton
convergence alone. dt grows only when `prev_delta/delta > 1` (the SER ratio is
shrinking — physics is approaching steady state). **Correct SER semantics.**

Floor at `dt_init` for the shrink branch is intentional: prevents dt collapse to
zero, but also means dt can't shrink below the initial probe size. **Acceptable
for the production SER use case.**

LOCATION: `grid_per_voltage.py:199-207`.

---

## Q11 — U_prev update inside SER loop (PRIORITY C)

**SEVERITY: NONE (correctly placed after Newton solve, before next iter).**

Code: `grid_per_voltage.py:187-211`:

```
for _ in range(1, max_steps + 1):
    try:
        solver.solve()                       # Newton on current U
    except Exception:
        return False
    U_prev.assign(U)                          # IMMEDIATELY after solve
    fv = float(fd.assemble(of_cd))
    if prev_flux is not None:
        ...steady-detect, dt update...
    prev_flux = fv
    if steady_count >= ss_consec:
        return True
return False
```

`U_prev.assign(U)` at line 192 fires **after** every successful Newton solve and
**before** anything else (observable, dt update). Next loop iteration's
`solver.solve()` evaluates `(U - U_prev)/dt` with U_prev = last accepted U. **SER
pseudo-time integration is intact.**

If solve raises, the function returns False before U_prev is updated — so caller
seeing False can safely restore from a snapshot, and U_prev is one step behind U
(but that's irrelevant since the snapshot path will overwrite both via
`_restore_U(ckpt, U, U_prev)` which calls `U_prev.assign(U)` at line 109).

LOCATION: `grid_per_voltage.py:192`.

---

## Q12 — solve_grid_with_anchor source-pool selection ordering

**SEVERITY: NONE (deterministic; documented behavior).**

Code: `grid_per_voltage.py:1078-1081`:

```
src_phi, src_snap = min(
    sources,
    key=lambda s: abs(float(s[0]) - target_phi),
)
```

Python's `min` returns the **first** element with the minimal key. `sources` is
built in append order (line 1048: anchor first; each successful target then
appended at line 1109). So if two sources are equidistant from `target_phi`, the
older (lower-index) source wins. **Deterministic. Acceptable.**

LOCATION: `grid_per_voltage.py:1078-1081`.

---

## Q13 — per_point_callback timing

**SEVERITY: NONE (called only on convergence — guarded by `if ok:`).**

Code: `grid_per_voltage.py:1107-1113`:

```
if ok:
    snap = _snapshot_U(U)
    sources.append((target_phi, snap))
    method = f"warm<-{src_phi:+.3f}"
    converged = True
    if per_point_callback is not None:
        per_point_callback(orig_idx, target_phi, ctx)
else:
    snap = None
    method = f"warm<-{src_phi:+.3f}-FAILED"
    converged = False
```

Callback only invoked inside the `if ok:` branch. ctx is in the **converged warm-
walked state** when the callback runs — the demo's plotting / per-point observable
extraction is safe. **Correct.**

LOCATION: `grid_per_voltage.py:1107-1113`.

---

## Q14 — observables._build_bv_observable_form mode dispatch

**SEVERITY: NONE (unrecognized → explicit ValueError).**

Code: `observables.py:99-163`. Dispatch is an if/elif chain on `mode_norm`
(normalized via `str(mode).strip().lower()`), terminating in an `else` raising
`ValueError(f"Unknown BV observable mode '{mode}'...")` (line 160-163). **Correct.
No silent default.**

LOCATION: `observables.py:159-163`.

---

## Q15 — n_e/N_ELECTRONS_REF=2 weighting & sign chain

**SEVERITY: NONE (documented, demonstrably consistent).**

Code: `observables.py:106-121`. `current_density` mode returns:

```
∫ scale * Σ_j (n_e_j / 2) * R_j ds(electrode_marker)
```

Where R_j is the BV rate UFL expression (signed by k0_anodic - k0_cathodic forms).
In the production stack R_j is positive for cathodic (reduction at cathode under
applied negative overpotential). The bare observable is therefore POSITIVE for
cathodic.

Demo applies `scale=-I_SCALE` (per `scripts/_bv_common.py:524, 533`) to flip sign
to the electrochemical convention (current density is conventionally negative for
cathodic faradaic current with the OUTWARD normal sign convention). This produces
the standard convention "negative current density at cathodic overpotentials" used
in the deck plotting code.

Chain: `R_j > 0 (cathodic)` × `n_e_j/2 > 0` × `-I_SCALE < 0` →
negative output for cathodic ORR. **Consistent.**

`N_ELECTRONS_REF = 2` matches the I_SCALE base; the per-reaction weight rescales
4e reactions by 2× and keeps 2e at 1×, so the assembled total `Σ (n_e/2)·R · I_SCALE`
gives physical current density. **Documentation matches code.**

LOCATION: `observables.py:43-121`.

---

## Additional findings (uncovered during read)

### A1 — `set_reaction_k0_model` `>= rxn_count` vs `>= len(rxns)`

`anchor_continuation.py:286-297` validates against `nondim.bv_reactions` length AND
`bv_k0_funcs` length separately. Both must match. If `bv_reactions` and `bv_k0_funcs`
drift in length (build-time bug elsewhere), this would catch it. **Defensive +
correct.**

### A2 — Picard-Γ outer loop early-break at λ=0 (anchor_continuation.py:1156)

In `_run_gamma_picard`, after `update_gamma_from_solution(ctx)` returns, the code
checks `lam_active == 0.0` and breaks. But Γ has already been written to ctx by
the update call. At λ=0, `update_gamma_from_solution` is documented to return
Γ=0 (per the docstring at line 1142). So Γ ends up at 0 (correct), and gamma_history
gets one entry. **Correct semantics; minor: at λ=0 the formula write is redundant
but not harmful** (the explicit `ctx["cation_hydrolysis"].gamma_func.assign(0.0)`
already happened at line 1366 in the floor branch).

### A3 — `_run_k0_ladder` catches `RuntimeError` (line 1229) for "AdaptiveLadder
misuse"

The ladder raises `RuntimeError` from `record_success`/`record_failure_and_insert`
when called past `is_done()` (`anchor_continuation.py:831-832`, `858-859`). The
try/except in `_run_k0_ladder` (line 1191/1229) treats this as ladder-exhausted,
not crash. The while-loop guard `while not k0_lad.is_done()` *should* preclude
ever hitting these RuntimeErrors. Defensive net — **acceptable**, but masks
genuine logic bugs if the loop guard is ever broken. Low priority.

### A4 — `phi_applied_func` is read at form-build via SolverParams.phi_applied
but the anchor-continuation path mutates it via `paf.assign(...)` AFTER the form is
built

This is the established pattern (paf is a Constant, not a value baked into the form
tree). Confirmed by hard rule + the `_clone_params_with_phi` helper
(`grid_per_voltage.py:994-999`). Each new ctx in `_build_for_voltage` rebuilds the
form against the cloned sp_v's phi — but warm_walk_phi then drives paf separately.
The form refers to paf by reference, so paf.assign() inside warm_walk_phi modifies
what the residual sees. **Correct (standard Firedrake idiom).**

### A5 — Stage 2's solver `ctx['_last_solver']` could be misused if a caller
explicitly extracted it across stages

There is no such caller in the audit scope. Stage 2's solver becomes unreachable
once Stage 2's ctx leaves scope. The dict entry exists; nothing in `_run_k0_ladder`,
`solve_grid_with_anchor`, or `warm_walk_phi` reads `ctx['_last_solver']` — they
all receive `solver` by parameter. **No real bug; potential foot-gun if someone
adds a downstream reader.**

---

## VERDICT

**No correctness bugs found in the audited code paths.**

All 15 priority questions resolve to either "correct as written" or "documented
edge case with proper guard."

Highest-priority deep traces:

- **Q4 (cross-stage solver leak):** Verified clean — `_build_for_voltage`
  constructs a fresh `NonlinearVariationalProblem` + `NonlinearVariationalSolver`
  per grid voltage against a freshly-built ctx. Stage 2's solver becomes
  unreachable.
- **Q7 (warm_walk_phi state restoration):** Verified the ckpt_outer/ckpt_inner +
  explicit `paf.assign(v_prev_substep)` pattern correctly maintains state
  invariants across failures and recursion entry/exit. The in-source comment
  (lines 272-278) accurately describes the requirement.
- **Q11 (U_prev update timing):** `U_prev.assign(U)` fires immediately after
  every successful `solver.solve()` (line 192) — SER is intact.

Minor non-blocking issues for future hardening:

- **Q3:** `sp[10]` positional read at `anchor_continuation.py:1095` is fragile to
  upstream SolverParams refactoring; consider a named accessor.
- **A3/A5:** Defensive RuntimeError handling and the unused `_last_solver` slot
  are foot-guns but not active bugs.

**Severity summary:**
- NONE: Q1 (effectively), Q2, Q4, Q5, Q6, Q7, Q8, Q11, Q12, Q13, Q14, Q15
- LOW: Q1 edge case, Q9, Q10
- MEDIUM: Q3 (refactor hazard, not current bug)

Overall: **PASS** on the audited code paths.
