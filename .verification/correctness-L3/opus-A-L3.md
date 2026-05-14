# Code Correctness Audit (L3, Opus, Third Pass)

Scope: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`,
`scripts/_bv_common.py`, `Forward/bv_solver/dispatch.py`,
`Forward/bv_solver/mesh.py`, `Forward/bv_solver/nondim.py`, plus
targeted reads of the call sites those files invoke
(`anchor_continuation.py`, `grid_per_voltage.py`, `observables.py`,
`params.py`, `config.py`, `forms_logc_muh.py`).

Format: SEVERITY / LOCATION / TRIGGER / EVIDENCE per issue.

---

## Findings

### F1. LOW — Stale `cd_mA_cm2` / `pc_mA_cm2` reported when Stage-3 convergence diagnostic disagrees with `_grab`

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
lines 517-565 (Stage-3 result assembly).

TRIGGER: A grid point where `warm_walk_phi` returns `ok = True` from
`solve_grid_with_anchor`, the demo's `_grab` callback records finite
values into `cd_arr[i]` / `pc_arr[i]`, then later the post-solve
`collect_diagnostics` call inside `solve_grid_with_anchor` (lines
1126-1134) raises — diagnostics are stored as `"diagnostics_error"`
but `converged` STAYS `True`. Or, the inverse: `ok=True` so callback
runs, but per-point convergence reporting is later flipped.

EVIDENCE: `solve_grid_with_anchor` sets `converged` from the
`warm_walk_phi` return BEFORE invoking the diagnostics block (lines
1107-1117). The callback is invoked only on `ok=True` (line 1113).
The reported `converged[i]` matches the value at callback time.
`_to_json_list` (demo line 560) gates on `converged[i]` AND
`np.isfinite(x)`. So if a callback fired but cd_arr[i] is NaN
(e.g. `_grab` exception caught), the JSON will record `None` even
though `converged[i]=True`. Reverse: if cd_arr is finite but
`converged[i]=False`, JSON records `None`. Both paths fail-closed.

No bug — fail-closed by design. Documented here for completeness.

---

### F2. LOW — `_grab` callback's exception swallowing produces silent NaN with no further annotation

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
lines 520-538.

TRIGGER: `_build_bv_observable_form` or `fd.assemble` raises during
callback (e.g. ctx missing `bv_rate_exprs`, or assemble divergence
on a marginal point). `_grab` prints the traceback class+message
and leaves `cd_arr[orig_idx]=NaN` / `pc_arr[orig_idx]=NaN`. The
emitted iv_curve.json then has `cd_mA_cm2[i]=None` while
`converged[i]=True` — i.e. the run claims convergence but yields
no observable.

EVIDENCE: lines 526-538: both `try` blocks catch broad `Exception`,
print, return without re-raising. No "had-error" flag is recorded
in the JSON. Downstream consumers (plotter) can detect `None` but
cannot distinguish "Newton converged but assemble failed" from
"Newton diverged". Minor reporting gap. Suggested: stash an
error-flag list alongside `converged` so consumers can disambiguate.

---

### F3. LOW — `_stern_bump_ladder(target)` truncates intermediate rungs when target lies *between* verified rungs

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
lines 304-322 (function body).

TRIGGER: A user supplies `--stern-final 0.30` (between verified
rungs 0.20 and 0.50). The ladder returns `[0.20, 0.30]` instead of
the (correct) `[0.20, 0.30]`. Actually correct in this case — but
the structure of the loop means it inserts target as the next
entry once a rung exceeds target, *without first inserting the
last good rung below target if that rung wasn't already appended*.

EVIDENCE: trace for `target=0.30`:
- rung=0.20: `0.20 >= 0.30`? False. `rungs.append(0.20)` → `[0.20]`.
- rung=0.50: `0.50 >= 0.30`? True. `rungs.append(0.30)`, return
  `[0.20, 0.30]`.

OK in this case. But for `target=0.15` (between STERN_ANCHOR=0.10
and first rung 0.20):
- rung=0.20: `0.20 >= 0.15`? True. `rungs.append(0.15)`, return
  `[0.15]`.

So we jump from 0.10 (anchor) to 0.15 in one shot. Bracket is fine
(1.5x growth). For `target=0.11`:
- rung=0.20: `0.20 >= 0.11`? True. `rungs.append(0.11)`, return
  `[0.11]`.

Single step 0.10→0.11. Also fine.

Edge case: `target = STERN_ANCHOR = 0.10` exactly — `target <=
STERN_ANCHOR` so returns `[0.10]`. Stage 2 then calls
`set_stern_capacitance_model(ctx_anchor, 0.10)` followed by
`solver.solve()`. The anchor was already built at C_S=0.10, so this
is a redundant no-op re-solve. Wastes a Newton call; not a bug.

Edge case: `target < STERN_ANCHOR` (e.g. `--stern-final 0.05`) →
`[0.05]`. Single down-bump from 0.10 to 0.05. May or may not
converge in Newton; depends on the Newton basin for that direction.
Not validated by the verified ladder.

Verdict: F3 is a documentation / sharp-edge issue, not a correctness
bug for the in-script default of `target ∈ {0.20, 100.0}`.

---

### F4. LOW — `factor_label` collisions for very-close factor values via `{:g}` formatting

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
line 247-248: `f"factor_{factor:g}"`.

TRIGGER: A user passes `--factors 1.0000001,1.000001` — Python
`{:g}` defaults to 6 significant digits, so both would round to
`"1.00000"`. The two factors then map to the **same** output
subdirectory; the second overwrites the first's `iv_curve.json`.

EVIDENCE: Default 4 factors `{1.0, 1e-6, 1e-12, 1e-18}` give
distinct labels `"factor_1", "factor_1e-06", "factor_1e-12",
"factor_1e-18"`. No collision at the production defaults.

Mitigation suggestion: use `{:.10g}` (10 sig figs) to widen the
collision window from 6 to 10. Or hash the factor for the dir name.

Severity LOW because production factors are well-separated.

---

### F5. LOW — All-Stage-1-failed summary path has no explicit "all failed" sentinel

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
lines 740-758, 760-773 (factor loop + summary).

TRIGGER: All 4 factors fail at Stage 1 (anchor build). Each
`_run_one_factor` returns a dict with `"anchor": {"converged":
False, "error": <msg>}`, `n_converged=0`. The factor loop continues
to the next factor. `summary_per_factor` aggregates entries with
`anchor_converged=False, n_converged=0, n_total=NV`. The summary.json
is written.

EVIDENCE: tested logic by hand:
- Each failure returns the early-return dict at line 415-441.
- `summary_per_factor.append(...)` line 751-757 just reads the
  fields; works fine on the failure dict.
- summary.json (lines 760-773) is written unconditionally.

No bug — summary is well-formed even in all-failed case. A consumer
checking `n_converged > 0` for at least one factor would correctly
detect total failure. Could be slightly more user-friendly with
an exit code > 0 on all-failed, but this is a design choice.

---

### F6. INFO — `_make_sp` shadowing of CHATGPT Bikerman-counterion entries (Cs+/SO4) still uses A_DEFAULT defaults internally

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
lines 160-173, plus `scripts/_bv_common.py` definitions of the
DEFAULT_*_BOLTZMANN_COUNTERION_STERIC entries.

TRIGGER: The demo overrides the dynamic species' (O2, H2O2, H+)
`a_vals_hat` to physical Marcus-radii values (correctly, per Hard
Rule #7). The counterion entries `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC`
and `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC` use their own
`a_nondim = A_CSPLUS_HAT` / `A_SO4_HAT` which ARE physical (per the
constants at `scripts/_bv_common.py:705-708`).

EVIDENCE: Bikerman a_nondim is correct for both dynamic and
counterion species in this demo. Cross-checked against Hard Rule #7
in CLAUDE.md. No bug. Documented for completeness.

Sanity arithmetic verification of H+ physical a:
- `_a_nondim_from_radius_m(2.80e-10) = (4/3)π(2.80e-10)^3 * 6.022e23
  * 1.2 = 6.65e-5` (nondim).
- Bikerman c_max ≈ 1/a (nondim) = 15037; physical = 15037 * 1.2
  mol/m³ = 1.80e4 mol/m³ ≈ Hard Rule #7's "1.8e4 mol/m³".
  Matches ✓.

---

### F7. MEDIUM — `with_solver_options(new_opts)` is shallow-replace; mutating nested dicts in the new sp aliases the original

LOCATION: `Forward/params.py:178-180`
(`with_solver_options`); demo uses lines 238-242.

TRIGGER: `dataclasses.replace(self, solver_options=opts)` returns a
new frozen `SolverParams` with `solver_options=opts`. `opts` is the
exact dict reference the caller passed in. Frozen-ness applies only
to the outer dataclass fields, NOT to the dict's nested values. If
two SolverParams instances share a nested dict (e.g. `bv_bc`), a
mutation to one is visible from the other.

EVIDENCE: demo does:
```python
new_opts = dict(sp.solver_options)        # shallow copy
new_bv = dict(new_opts["bv_convergence"]) # shallow copy of inner
new_bv["exponent_clip"] = 100.0
new_opts["bv_convergence"] = new_bv       # replace inner reference
sp = sp.with_solver_options(new_opts)
```
This is correct: the inner `bv_convergence` is also shallow-copied
before mutation. Other inner dicts (`bv_bc`, `nondim`, etc.) are
shared by reference between OLD `sp` and NEW `sp`. **The demo never
mutates those, so no observable bug.** But the immutability invariant
("frozen=True ⇒ no aliasing") is violated at the nested level.

A subtle related concern: every call to `_make_sp` constructs a
fresh `sp` via `make_bv_solver_params(...)`, which builds a fresh
`params` dict from scratch (no shared inner state). So per-factor
the dicts are isolated. Cross-factor aliasing risk is zero given
the current call pattern.

Severity MEDIUM because the pattern is fragile: a future contributor
who writes `sp_anchor = sp_baseline.with_phi_applied(...)` then
mutates `sp_baseline.solver_options['bv_bc']['k0_targets']` would
silently mutate `sp_anchor` too — a deep-copy at `with_*` would
prevent this, at a small cost. See `deep_copy()` method
(params.py:150-164) which IS deep — but `with_solver_options`
isn't. Inconsistency.

Suggested fix: `with_solver_options(opts)` should deep-copy `opts`
on entry (or document the shallow-aliasing contract loudly).

---

### F8. LOW — `set_stern_capacitance_model(ctx, 0.0)` is permitted but the Stern Robin BC term becomes a zero-coefficient — silently equivalent to but NOT identical to no-Stern Dirichlet

LOCATION: `Forward/bv_solver/anchor_continuation.py:412-467`
(`set_stern_capacitance_model`).

TRIGGER: A caller passes `c_s_f_m2 = 0.0`. The setter validates
`c_s_f_m2 < 0.0` (line 444) only. At `c_s_f_m2 = 0.0`,
`stern_const.assign(0.0)` sets the Robin coefficient to 0, which
**does not collapse to** the no-Stern Dirichlet because the Robin
BC's form `eps_coeff * grad(phi).n = stern_const * (phi_m - phi)`
becomes `eps_coeff * grad(phi).n = 0` — a no-flux/Neumann BC, not
a Dirichlet BC. This is a different physics regime!

EVIDENCE: docstring (line 432-436):
> "the C_S = 0 limit is the no-Stern Dirichlet (build-time decision,
> not runtime), so this setter rejects ``< 0`` only."

The docstring acknowledges this and asserts that 0 IS rejected
**at build time**, which is correct for the script path (the demo
never calls `set_stern_capacitance_model(ctx, 0)`). But the setter
itself allows 0, with no warning. If a future caller paths in here
expecting `C_S = 0 → Dirichlet`, they'd get Neumann instead.

Severity LOW because no current call site passes 0, and the
docstring warns. Suggested: raise on `c_s_f_m2 == 0.0` with a
pointer to the no-Stern build-time path, or assert `> 0` rather
than `>= 0`.

---

### F9. LOW — `dispatch._read_bv_convergence_field` uses `solver_params[10]` magic index — fragile against future SolverParams schema changes

LOCATION: `Forward/bv_solver/dispatch.py:49`.

TRIGGER: SolverParams field layout in `params.py` adds a new field
at position 10 or earlier. The `solver_params[10]` index then points
to the wrong field, but `_read_bv_convergence_field` returns the
default silently.

EVIDENCE: line 49: `params_dict = solver_params[10] if hasattr(...)`.
The constant `10` is the historical position of `solver_options` in
the legacy 11-tuple layout. SolverParams's `_NAMES` tuple at
`params.py:58-62` puts `solver_options` at index 10 (confirmed).
`__getitem__` (params.py:86-88) uses `to_list()[index]` which mirrors
the same order.

If a contributor adds a field between `phi0` and `solver_options`
without updating `_N=11` and updating callers of `[10]`, the
dispatcher silently falls back to the default formulation `"logc"`.
That would be the safest fallback (production default) but masks the
schema-mismatch bug.

Suggested: use the named attribute `solver_params.solver_options`
when available, and fall back to index only for legacy 11-tuples.

Pattern is consistent across the codebase, so this is a global
fragility, not script-specific. Severity LOW.

---

### F10. LOW — `_a_nondim_from_radius_m` ignores `_C_SCALE` mismatch with `scripts/_bv_common.C_SCALE`

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
lines 107-113.

TRIGGER: `_C_SCALE = 1.2` is a private constant in the demo, used
to build `A_O2_PHYSICAL`, `A_H2O2_PHYSICAL`, `A_HP_PHYSICAL`. The
production-aligned `C_SCALE` lives in `scripts/_bv_common.py:133`
as `C_SCALE = C_O2 = 1.2`. They match today (May 2026). But the
demo's local constant means a future bump to `_bv_common.C_O2`
would not propagate; the demo's `A_*_PHYSICAL` would silently
drift from the production scale, breaking nondim consistency.

EVIDENCE: demo line 107: `_C_SCALE = 1.2`. Common-module:
`C_SCALE = C_O2 = 1.2` (lines 86 + 133). The demo could `from
scripts._bv_common import C_SCALE as _C_SCALE` to lock the two
together. Currently they're hardcoded duplicates.

Suggested: import C_SCALE from `_bv_common` rather than hardcoding.

---

### F11. INFO — Mesh grading sum for Ny=80, beta=3, D=1: y[80]=1.0 (FP-exact), monotonic, Firedrake-safe

ANSWER to Question A:
- `coords = mesh.coordinates.dat.data` returns one (x, y) per
  vertex, with y values in {0, 1/80, 2/80, …, 80/80}. After
  `coords[:, 1] = (coords[:, 1] ** 3) * 1.0`, y values become
  {(i/80)^3} for i=0..80.
- FP exactness: `(0/80)^3 = 0`. `(80/80)^3 = 1.0^3 = 1.0` (exact).
  `(40/80)^3 = 0.5^3 = 0.125` (exact). Intermediate values:
  `(1/80)^3 ≈ 1.953e-6` (smallest), `(79/80)^3 ≈ 0.9628` (next
  to top).
- Σ(Δy) for i=1..80 = y[80] − y[0] = 1.0 (telescoping). ✓
- Smallest Δy = y[1] − y[0] = 1.953e-6. Largest Δy = y[80] − y[79]
  = 1 − 0.9628 = 0.0372. Ratio ≈ 19,000x.
- Firedrake `fd.RectangleMesh(Nx, Ny, 1.0, 1.0)` accepts any
  monotonic remap of `coords[:, 1]` provided each remapped value
  stays in the convex hull of original (so element Jacobians stay
  positive). Since y^3 is monotonic on [0, 1] with strictly positive
  derivative, Jacobians remain positive. ✓
- MPI safety: `mesh.coordinates.dat.data` accesses local-rank data.
  The remap is local and consistent because (x, y) of any global
  vertex is invariant; each rank applies y → y^3 to its share, the
  halo sync makes shared vertices agree. ✓

No bug.

---

### F12. INFO — `bv_reactions` legacy-bundle silent ignore

LOCATION: `Forward/bv_solver/forms_logc_muh.py:520-636`,
`Forward/bv_solver/nondim.py:104-167`.

CLAUDE.md hard rule warns:
> Don't pass `bv_reactions=...` with `k0_hat_r1`/`k0_hat_r2`/etc.
> (reactions list takes precedence; legacy bundle silently ignored).

The demo's `_make_sp` passes `bv_reactions=rxns` but does NOT
explicitly pass `k0_hat_r1`/`k0_hat_r2`. They default to
`K0_HAT_R1`/`K0_HAT_R2` in `make_bv_solver_params`'s signature
(lines 1072-1073), and are then written into `bv_bc` config via
`_make_bv_bc_cfg`. The forms code then dispatches by the presence
of `bv_reactions` (the reactions list). Since they coexist but the
reactions list takes precedence, the legacy values are harmless
ignored — exactly as documented. No bug.

But: the script could explicitly pass `k0_hat_r1=0.0, k0_hat_r2=0.0`
to make the legacy bundle obviously inert. Cosmetic.

---

### F13. LOW — `solve_grid_with_anchor` `params` reference is a STATIC dict captured at `_run_one_factor` time; later `set_*_model` mutations on the LIVE ctx do not reflect back

LOCATION: `Forward/bv_solver/grid_per_voltage.py:984` and
`1023-1024`.

TRIGGER: `solve_grid_with_anchor` reads `params = solver_params[10]`
ONCE (line 984), captures it, and uses it to derive `solve_opts`
(line 1023-1024) for each per-voltage Newton solve. If any prior
phase has used `set_stern_capacitance_model` or `set_reaction_k0_model`
on the anchor's ctx, those mutations live on the OLD ctx, not on
the captured `params` dict. Each per-voltage build (`build_context
+ build_forms`) rebuilds from `params` (which still has the
Stage-1-build-time values), then `set_reaction_k0_model` is invoked
to re-pin k0 (lines 1016-1022). But Stern is NOT re-pinned from
the anchor — only k0.

EVIDENCE: demo Stage 2 uses `set_stern_capacitance_model(ctx_anchor,
stern_final_v)` to bump the anchor's live Stern. At Stage 3,
`solve_grid_with_anchor(sp_baseline, ...)` is called with
`sp_baseline` built at `stern_capacitance_f_m2=stern_final_v`
(line 376-379). So `sp_baseline.solver_options[...]['bv_bc']`
already has C_S=stern_final_v baked in at build time. ✓

The Stage-3 per-voltage rebuild uses `sp_baseline` and gets the
correct C_S directly from the config. The anchor's runtime
Stern-bump doesn't matter — what's relevant is what's in the
SolverParams the grid driver was handed.

No bug. Documented to confirm the demo's two-instance design
(separate `sp_anchor_cs` at C_S=0.10 and `sp_baseline` at
C_S=0.20) is the correct way to handle this.

---

### F14. INFO — adj.stop_annotating lifecycle is correct

ANSWER to Question D:
- Stage 1 wraps `solve_anchor_with_continuation` in
  `adj.stop_annotating()` (demo line 397-404).
- Stage 2: each iteration wraps `ctx_anchor["_last_solver"].solve()`
  in `adj.stop_annotating()` (demo line 457-458).
- Stage 3: `solve_grid_with_anchor` wraps its own loop in
  `adj.stop_annotating()` (grid_per_voltage.py:1060).
- Between Stage 2 exit and Stage 3 entry, lines 499-552 run with
  the tape ON, BUT the only operations are:
  * `snapshot_U(ctx_anchor["U"])` — accesses `d.data_ro.copy()` at
    the PETSc DAT level. NOT a `fd.Function` op; not recorded by
    pyadjoint.
  * `tuple(np.asarray(arr).copy() for arr in U_post_bump)` — pure
    NumPy. Not recorded.
  * `PreconvergedAnchor(...)` — dataclass construction. Not recorded.
  * `cd_arr = np.full(NV, np.nan)` / `pc_arr = np.full(NV, np.nan)`
    — pure NumPy.
  * `np.array(V_RHE_GRID, ...) / V_T` — pure NumPy.
- No `fd.Function`-typed operations cross the tape between Stage 2
  and Stage 3 entry. ✓

No bug.

---

### F15. INFO — `K0_R4E_FACTORS` loop ctx isolation

ANSWER to Question E:
- Each factor calls `_make_sp(...)` twice (anchor + baseline). Each
  produces a fresh frozen `SolverParams` with a fresh `params` dict
  (in `make_bv_solver_params`, line 1251: `params = dict(snes_opts or
  SNES_OPTS)` — new dict every call).
- Each factor calls `solve_anchor_with_continuation` which calls
  `build_context` then `build_forms`. `build_forms` creates new
  `fd.Function(R_space, name=f"bv_k0_rxn{j}")` per reaction — the
  names are identical across factors but each Function lives on a
  freshly-created `R_space` (a new `fd.FunctionSpace` per ctx build).
- Firedrake's kernel cache is keyed by UFL form structure + element
  type, NOT by Function names; name collisions across factors are
  benign for the cache (in fact beneficial — same kernel reused).
- No global module-level Constants whose value crosses factors.
  All `fd.Constant(...)` are local to a single ctx build.

No bug. MPI safety also confirmed — no module-level mutable state.

---

### F16. INFO — `_grab` callback exception-safety against `fd.assemble`

ANSWER to "Is _grab exception-safe?":
- `_grab` wraps both observable assemblies in `try/except Exception`
  (demo lines 526-538). Any raised exception is caught, the
  traceback class+message is printed, the corresponding array slot
  stays NaN.
- The calling `solve_grid_with_anchor` invokes the callback
  unprotected (line 1112-1113), but `_grab` itself never re-raises.
- SystemExit/KeyboardInterrupt are NOT caught by `except Exception`
  (they derive from `BaseException`); a Ctrl-C during `_grab` would
  propagate. Expected behavior.

No bug.

---

### F17. INFO — Multi-process / parallel safety

ANSWER to "MPI Constants aliasing?":
- No `fd.Constant(...)` at module scope in
  `solver_demo_slide15_no_speculative_cs.py` or its imports'
  module-init paths. All `fd.Constant` instances are created inside
  `build_context_logc_muh` / `build_forms_logc_muh`, per-ctx.
- `THREE_SPECIES_LOGC_BOLTZMANN` is a frozen dataclass (no Firedrake
  state). Same for `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` —
  it's a plain dict of plain Python floats.
- Mesh is shared across factors but is a Firedrake parallel object;
  reuse is intentional and safe.

No MPI aliasing risk in the demo path.

---

### F18. LOW — `_to_json_list` uses `converged[i]` as a gate, but `converged` is captured from `grid_result.points[i].converged` — assumes `points` is dense over `[0, NV)`

LOCATION: `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
lines 554-555.

TRIGGER: `solve_grid_with_anchor` returns `points: Dict[int,
PerVoltagePointResult]`. The demo dereferences `grid_result.points[i]
for i in range(NV)`. If a `i` is missing from the dict (e.g. an
internal exception aborted the grid walk mid-loop), `KeyError`
fires inside the list-comprehension and the entire `_run_one_factor`
call raises uncaught.

EVIDENCE: `solve_grid_with_anchor` iterates `visit_order` for all
`n_points = len(phi_applied_values)` entries (grid_per_voltage.py:1061),
populating `points[orig_idx]` unconditionally at line 1136. Only an
exception inside `_build_for_voltage` or `warm_walk_phi` could skip
a `points[i]` assignment — and the only `raise` inside
`_build_for_voltage` is the `ValueError` at line 1070 for mesh DOF
mismatch (a programmer error, fail-fast).

So practical risk is zero unless DOF mismatch happens or the loop
itself raises. But the demo could iterate `for i in
grid_result.points` to be defensive against future driver changes.

Severity LOW.

---

## Question Answers Summary

- **A**: Mesh grading FP-exact and Firedrake-safe (F11).
- **B**: Physical a_nondim arithmetic matches Hard Rule #7 (F6).
- **C**: `with_solver_options` uses `dataclasses.replace`, returns a
  new frozen instance, but does NOT deep-copy the nested dict —
  aliasing risk at the nested-dict level (F7). Cost per call is one
  dataclass alloc + one dict reference — negligible.
- **D**: adj.stop_annotating lifecycle is correct, no recorded
  operations escape (F14).
- **E**: Per-factor ctx isolation is clean (F15).
- **F**: `_stern_bump_ladder` edge cases — `target<=anchor` returns
  `[target]`, `target` between rungs returns `[…, target]` without
  inserting the bracketing upper rung. Documented (F3).
- **G**: `factor_label` distinct for production factors {1, 1e-6,
  1e-12, 1e-18} (F4).
- **H**: `dispatch` silently falls through unknown formulation to
  `"logc"` (default). The config layer
  (`config.py:_validate_formulation`) DOES reject unknown names at
  parse time with `ValueError`. So at runtime, unknown formulations
  cannot reach the dispatcher unless someone bypassed the validator.
  Dispatcher is "defense in depth" — F9 minor fragility.
- **I**: See F1–F18.

NEW questions:
- **SolverParams construction validation**: `_validate_formulation` /
  `_validate_initializer` reject typos at parse time
  (`config.py:82-117`). Good. But these run inside `_get_bv_convergence_cfg`,
  which is invoked lazily when forms are built, NOT at SolverParams
  construction. A typo in `formulation="loooogc_muh"` would
  silently accept the SolverParams object and fail only on
  `build_context`. Modest defense-in-depth gap.
- **_grab exception safety**: Yes, robust (F16).
- **MPI safety**: Yes, no global Constants alias across ranks (F17).
- **Summary output on all-fail**: Yes, well-formed (F5).

---

## VERDICT

No correctness-blocking bugs identified in the demo path. The
production-default sweep (4 factors {1, 1e-6, 1e-12, 1e-18} × 25
V_RHE points) is safe to run as written.

Material findings worth tracking:
1. **F7 (MEDIUM)**: `with_solver_options` is shallow-replace.
   Demo's call pattern is safe today but the immutability invariant
   is violated at the nested-dict level. Suggest a deep-copy or a
   loud doc-string warning. The codebase already has `deep_copy()`
   for cases that need full isolation — making `with_*` consistent
   would tighten the invariant.

2. **F8 (LOW)**: `set_stern_capacitance_model(ctx, 0.0)` silently
   produces a no-flux Neumann BC, NOT a Dirichlet. The docstring
   warns; production code never hits this branch.

3. **F10 (LOW)**: `_C_SCALE = 1.2` is hardcoded in the demo;
   diverging from `_bv_common.C_SCALE` would silently break
   nondim consistency.

4. **F2 (LOW)**: `_grab` callback swallows exceptions silently with
   only a print. A future post-mortem on `iv_curve.json` cannot
   distinguish "Newton OK but observable assemble failed" from
   "Newton diverged". Suggest a per-point error-flag list.

5. **F4 (LOW)**: `{:g}` formatting collapses similar-magnitude
   factors into the same directory name. Production factors are
   well-separated; not a bug today.

All other items (F1, F3, F5, F6, F9, F11–F18) are either INFO
(no-bug confirmations) or LOW-severity sharp edges with no
observable failure mode at the production defaults.

**Recommendation: PASS with optional cleanup items F2, F7, F10.**
