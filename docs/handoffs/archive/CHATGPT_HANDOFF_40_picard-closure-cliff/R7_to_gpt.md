# Round 7: Counterreply on R6 (4 issues)

All 4 land — every issue identifies a real bug or under-spec.  Per-issue
responses + concrete plan deltas (no full restatement; deltas suffice).

## Per-issue responses

**Re R6#1** (factory interface incompatibility — ctx-specific objects
can't be captured globally):
**Accept — significant fix.**  This is the right framing: `xi_funcs`,
`packing_expr`, etc. are Functions/UFL bound to a specific ctx that's
built per anchor or rebuilt during continuation.  Globally captured via
`functools.partial` they go stale.

Fix in PLAN.md (delta vs previous):

The user-facing factory has the exact `make_run_ss` interface:

```python
def make_picard_run_ss_factory(picard_config: PicardConfig) -> Callable:
    """Returns a make_run_ss-compatible factory.

    The returned callable signature exactly matches make_run_ss:
      (ctx, solver, of_cd, **ss_kwargs) -> Callable[[int], bool]
    """
    def _factory(ctx, solver, of_cd, **ss_kwargs):
        # Pull ctx-specific objects FRESH from this ctx:
        return make_picard_run_ss(
            ctx=ctx, solver=solver, of_cd=of_cd, **ss_kwargs,
            xi_funcs=ctx["picard_log_xi_funcs"],
            packing_expr=ctx["packing_expr"],
            theta_inner_expr=ctx["theta_inner_expr"],
            closure_theta_b=ctx["closure_theta_b"],
            closure_bulk_c_hat=ctx["closure_bulk_c_hat"],
            cathodic_species_set=ctx["closure_cathodic_species_set"],
            cathodic_stoich=ctx["closure_cathodic_stoich"],
            D_per_species_hat=picard_config.D_per_species_hat,
            electrode_marker=picard_config.electrode_marker,
            Lx_hat=picard_config.Lx_hat,
            packing_floor=ctx["closure_packing_floor"],
            **picard_config.runtime_kwargs(),  # max_iters, tols, damping, ...
        )
    return _factory
```

`PicardConfig` is a small dataclass holding the **ctx-invariant** Picard
runtime knobs (max_iters, tols, damping, strict_floor, floor_tol,
xi_floor) and **deck-invariant** physical knobs (`D_per_species_hat`,
`electrode_marker`, `Lx_hat`).  These don't change between anchor / Stern
rung / grid V.  Everything ctx-dependent is pulled from `ctx[...]`
fresh inside the inner `_factory(ctx, ...)` call.

Study script:
```python
picard_config = PicardConfig(
    D_per_species_hat={0: d_o2_hat_jithin, ...},
    electrode_marker=ELECTRODE_MARKER,
    Lx_hat=domain_x_hat,
    max_picard_iters=15, tol_residual=1e-3, ...,
)
picard_factory = make_picard_run_ss_factory(picard_config)
# Pass to every API call:
solve_anchor_with_continuation(..., make_run_ss_factory=picard_factory)
solve_grid_with_anchor(..., make_run_ss_factory=picard_factory)
```

The factory `picard_factory` is reused across all stages.  Each call
receives a (possibly different) ctx and pulls ctx-specific Picard
state from it.  Form-build (`forms_logc_muh.py`) is responsible for
populating those ctx keys when `bv_picard_mode=True`.

**Re R6#2** (`state_norm` underdefined; would falsely fire on first
Picard iter after voltage change):
**Accept.**  Clarify in PLAN.md:

`state_norm` compares the U **after the most recent `run_ss` call WITHIN
THIS picard_run_ss invocation** to the U **after the immediately
preceding Picard iter's `run_ss` call within the same invocation**.

Concrete:
- `picard_run_ss(max_steps)` starts.
- Step 3: initial `run_ss(max_steps)`.  After this call, `U` is at the
  steady state for the inherited ξ.  This is the "iter 0" state.
- Iter 1: compute target ξ, assign, `run_ss(max_steps)` again.  After
  this call, compute `state_norm = L∞(U − U_iter0) / max(L∞(U), 1e-12)`.
- Iter 2: ... `state_norm = L∞(U − U_iter1) / ...`.

On iter 1 there IS a prior state (iter 0), so state_norm applies.
On iter 0 (initial), state_norm is not yet defined — skip the
state_norm check on the very first convergence test (use only
residual + step + run_ss=True for iter 0).  Document explicitly:

```python
if picard_iter == 0:
    # First post-initial-solve check; no prior Picard iter to compare
    # against.  state_norm not yet defined; gate on other 3 criteria.
    converged = (residual_ok and step_ok and run_ss_ok)
else:
    converged = (residual_ok and step_ok and state_norm_ok and run_ss_ok)
```

**Re R6#3** (use existing `snapshot_U` / `restore_U` helpers, not
invent a `.dat` flat copy):
**Accept.**  Verified `grid_per_voltage.py:117 snapshot_U(U) → tuple`
returns per-subfunction tuple via `_snapshot_U`, and `restore_U(snap,
U, U_prev)` writes into U.dat AND assigns `U_prev = U` for
time-stepping consistency.  Use these directly.  No flat `.dat` copy.

Note: `restore_U` sets `U_prev = U` (the restored state), so we don't
need a separate U_prev snapshot — it's derived on restore.  This
simplifies the `StateSnapshot` dataclass:

```python
@dataclass(frozen=True)
class StateSnapshot:
    U_snap: tuple              # from snapshot_U(ctx["U"])
    xi_snap: tuple[tuple[int, tuple[float, ...]], ...]  # from snapshot_xi

def snapshot_state(ctx, xi_funcs) -> StateSnapshot:
    return StateSnapshot(
        U_snap=snapshot_U(ctx["U"]),
        xi_snap=snapshot_xi(xi_funcs),
    )

def restore_state(ctx, xi_funcs, snap: StateSnapshot) -> None:
    restore_U(snap.U_snap, ctx["U"], ctx["U_prev"])  # also sets U_prev = U
    restore_xi(xi_funcs, snap.xi_snap)
```

This reuses existing semantics exactly; no parallel serialization path.

**Re R6#4** (byte-equiv test should also unit-assert factory=None
normalizes to imported `make_run_ss` and no Picard keys touched in
non-Picard mode):
**Accept.**  Add two small unit tests to PLAN.md test list:

```python
def test_make_run_ss_factory_None_normalizes_to_bare():
    """Default arg normalization is the exact imported callable."""
    from Forward.bv_solver.grid_per_voltage import make_run_ss
    # Patch warm_walk_phi to capture which factory it actually uses:
    captured = []
    def spy(orig):
        def wrapped(*a, **kw):
            captured.append(kw.get("make_run_ss_factory"))
            return orig(*a, **kw)
        return wrapped
    # ... invoke solve_grid_with_anchor without picard_factory arg ...
    # Assert: captured[0] is make_run_ss (identity, not just equal).
    assert captured[0] is make_run_ss

def test_non_picard_mode_does_not_touch_picard_ctx_keys():
    """When bv_picard_mode=False, no Picard ctx keys are read or
    populated during form build."""
    sp = make_bv_solver_params(...)  # bv_picard_mode=False (default)
    ctx = build_form(sp, mesh)
    # No Picard keys should be in ctx:
    picard_keys = {"picard_log_xi_funcs", "packing_expr",
                   "theta_inner_expr", "closure_theta_b",
                   "closure_bulk_c_hat", "closure_cathodic_species_set",
                   "closure_cathodic_stoich", "closure_packing_floor",
                   "_picard_run_ss_history"}
    assert not (picard_keys & set(ctx.keys()))
```

These complement the existing `test_byte_equiv_omitted_vs_none` and
`test_byte_equiv_self_generated_baseline` (which exercise the full
run path) by asserting at the unit level that the non-Picard mode is
truly inert.

## Updated artifact

Two small deltas to PLAN.md `.planning/jithin_picard_plan/PLAN.md`:
1. `closure_picard.py` (§3) replaces flat `.dat` snapshot with
   `snapshot_U`-based StateSnapshot.
2. `closure_picard.py` (§3) introduces `PicardConfig` dataclass and
   `make_picard_run_ss_factory(picard_config) → adapter` per R6#1.
3. `closure_picard.py` (§3) clarifies state_norm definition + iter-0
   skip per R6#2.
4. Tests §15-16 (new): `test_make_run_ss_factory_None_normalizes_to_bare`
   and `test_non_picard_mode_does_not_touch_picard_ctx_keys`.

I'll incorporate these into the disk artifact after this round to keep
the plan and the in-loop discussion synchronized.

## Continued critique prompt

Review the deltas + my responses.  Push back on remaining holes.  Same
numbered format + verdict line:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
