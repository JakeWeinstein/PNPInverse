# Critique session 39 — Round 6

(Loop extended from cap=5 to cap=7. R5 issues are real and your
"cap reached" framing turned out to be premature — keep pushing.)

## Section 1: Acknowledgment

All 5 R5 issues accepted. They're a coherent cluster around
**ownership-tracking and ordering** of the independence check, which
the plan had hand-waved.

### Re point 1 — independence check ordering

**Accept.** Source terms `S_c_i`, `g_i`, `S_φ`, `g_S` only exist after
construction inside `_build_manufactured_source`. So an independence
check cannot fire in `_assert_runtime_stack_invariants` before sources
exist. Two-stage source construction:

```python
@dataclass
class MMSSourceTerms:
    """Local container for the four source UFL expressions, built BEFORE
    F_res mutation so they can be inspected by the independence check."""
    S_c: list                # one per dynamic species (UFL)
    g_elec: list             # one per dynamic species (UFL, on ds_elec)
    S_phi: object            # UFL (on dx)
    g_S: object              # UFL (on ds_elec)


def _build_source_terms(
    ctx, sp, *, manuf, closure, rxn_rates, owned: OwnedCoeffTracker,
    quad_degree: int,
) -> MMSSourceTerms:
    """Pure construction: compose source UFL expressions WITHOUT mutating
    F_res. Returns the container; F_res mutation happens later in
    _inject_source_terms after the independence check passes."""
    ...
    return MMSSourceTerms(...)


def _assert_source_independence(sources: MMSSourceTerms, ctx: dict,
                                 owned: OwnedCoeffTracker) -> None:
    """Inspect each source UFL expression. Forbidden: ctx['U'],
    ctx['U_prev']. Allowed: live ctx coefficients + owned literals.
    Anything else is unknown → AssertionError."""
    ...


def _inject_source_terms(ctx: dict, sources: MMSSourceTerms, *,
                          quad_degree: int) -> dict:
    """Mutate ctx['F_res'] by subtracting each source, then rederive J_form."""
    ...


def _build_manufactured_source(ctx, sp, *, manuf, closure, rxn_rates,
                                quad_degree, owned):
    sources = _build_source_terms(ctx, sp, manuf=manuf, closure=closure,
                                   rxn_rates=rxn_rates, owned=owned,
                                   quad_degree=quad_degree)
    _assert_source_independence(sources, ctx, owned)
    return _inject_source_terms(ctx, sources, quad_degree=quad_degree)
```

The runtime invariant harness (`_assert_runtime_stack_invariants`) no
longer tries to check source independence — it covers the margin
checks only (clamps, R_ratio, θ_inner). Source independence is its own
phase, run BETWEEN `_build_source_terms` and `_inject_source_terms`.

### Re point 2 — owned-constant tracker threads through all builders

**Accept.** Replaced the module-global `_OWNED_COEFFS` set with an
explicit per-mesh tracker that every MMS expression builder receives:

```python
class OwnedCoeffTracker:
    """Per-mesh ledger of fd.Constant objects created by MMS builders.
    Threaded through all expression-construction helpers so the
    independence check has a complete whitelist by source identity."""

    def __init__(self):
        self._owned: set[fd.Constant] = set()

    def constant(self, value: float, *, label: str = "") -> fd.Constant:
        c = fd.Constant(float(value))
        self._owned.add(c)
        return c

    @property
    def coeffs(self) -> set:
        return frozenset(self._owned)
```

Threaded into every builder signature:

```python
def _make_manufactured_fields(mesh, ctx, sp, *, owned: OwnedCoeffTracker) -> dict: ...
def _build_shared_theta_closure_ex(*, owned: OwnedCoeffTracker, ...) -> dict: ...
def _build_bv_rates_ex(*, owned: OwnedCoeffTracker, ...) -> list: ...
def _build_source_terms(..., *, owned: OwnedCoeffTracker, ...) -> MMSSourceTerms: ...
```

Every `fd.Constant(...)` call inside an MMS builder is replaced with
`owned.constant(...)`. `_assert_source_independence` consumes
`owned.coeffs` as the whitelist of MMS-owned literals.

### Re point 3 — no `isinstance(fd.Constant)` filter; identity-only

**Accept.** The whitelist is purely object-identity-based:

```python
def _assert_source_independence(sources, ctx, owned):
    FORBIDDEN = {ctx['U'], ctx['U_prev']}
    ALLOWED_LIVE = {
        ctx['phi_applied_func'],
        ctx['stern_coeff_const'],
        *ctx.get('bv_k0_funcs', []),
        *ctx.get('bv_alpha_funcs', []),
        ctx['boltzmann_z_scale'],
    }
    ALLOWED_LIVE.discard(None)
    ALLOWED = ALLOWED_LIVE | owned.coeffs

    for label, S in _iter_source_terms(sources):
        coeffs = set(extract_coefficients(S))
        bad = coeffs & FORBIDDEN
        assert not bad, (
            f"Source {label} references {bad} — independence violation"
        )
        unknown = coeffs - ALLOWED
        assert not unknown, (
            f"Source {label} references unrecognized coefficients {unknown}. "
            f"Either route through OwnedCoeffTracker or add to ALLOWED_LIVE."
        )
```

No `isinstance` filtering. Anything outside FORBIDDEN ∪ ALLOWED fails.

### Re point 4 — `OwnedCoeffTracker` per mesh, not module global

**Accept.** Tracker is created at the top of `run_mms` per-mesh
iteration and passed down:

```python
def run_mms(N_list, ...):
    results = {...}
    for N in N_list:
        mesh = fd.UnitSquareMesh(N, N)
        sp = make_sp_production_muh()
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        _assert_static_stack_invariants(ctx, sp, solver_params=sp)

        owned = OwnedCoeffTracker()   # fresh tracker per mesh
        manuf    = _make_manufactured_fields(mesh, ctx, sp, owned=owned)
        closure  = _build_shared_theta_closure_ex(..., owned=owned)
        rxn_rates= _build_bv_rates_ex(..., owned=owned)

        _assert_runtime_stack_invariants(ctx, sp,
                                          manuf=manuf, closure=closure,
                                          rxn_rates=rxn_rates,
                                          quad_degree=quad_degree)
        _build_manufactured_source(ctx, sp,
                                    manuf=manuf, closure=closure,
                                    rxn_rates=rxn_rates,
                                    quad_degree=quad_degree,
                                    owned=owned)

        U_manuf = _interpolate_U_manuf(ctx, manuf)
        ctx['U'].assign(U_manuf)
        ctx['U_prev'].assign(U_manuf)

        # snapshot live continuation values
        snapshots = _snapshot_live_coeffs(ctx)
        out_mesh = _solve_and_measure(ctx, sp, mesh, manuf)
        _assert_live_coeffs_unchanged(ctx, snapshots)
        _append_results(results, N, out_mesh)
    return results
```

No global state. Tests can run in any order, nested smoke tests can
spin up their own tracker.

### Re point 5 — explicit ordering statement (with diagram)

**Accept.** Added to PLAN.md §3 as a new "Build-time pipeline" subsection:

```
Per-mesh pipeline (left-to-right, top-to-bottom):

  build_context(sp)
      ↓
  build_forms(ctx, sp)
      ↓
  _assert_static_stack_invariants(ctx, sp)            ← formulation, reactions,
      ↓                                                  species, dt, SNES, etc.
  OwnedCoeffTracker()
      ↓
  manuf    = _make_manufactured_fields(mesh, ctx, sp, owned=owned)
  closure  = _build_shared_theta_closure_ex(..., owned=owned)
  rxn_rates= _build_bv_rates_ex(..., owned=owned)
      ↓
  _assert_runtime_stack_invariants(ctx, sp, manuf, closure, rxn_rates)
                                                      ← clip/floor margins,
      ↓                                                  R_ratio, θ_inner
                                                         (margin asserts MUST
                                                          pass BEFORE next step,
                                                          otherwise unclamped
                                                          source builder uses
                                                          wrong algebra)
  sources  = _build_source_terms(ctx, sp, manuf, closure, rxn_rates,
                                  owned=owned, quad_degree=quad_degree)
      ↓
  _assert_source_independence(sources, ctx, owned)    ← extract_coefficients
                                                         vs FORBIDDEN ∪ ALLOWED
      ↓
  _inject_source_terms(ctx, sources, quad_degree=quad_degree)
                                                      ← mutates F_res
      ↓
  U_manuf = _interpolate_U_manuf(ctx, manuf)
  ctx['U'].assign(U_manuf); ctx['U_prev'].assign(U_manuf)
      ↓
  snapshots = _snapshot_live_coeffs(ctx)
      ↓
  solver.solve()
      ↓
  _assert_live_coeffs_unchanged(ctx, snapshots)
      ↓
  compute errors per primary unknown
```

The **load-bearing ordering invariant**: `_assert_runtime_stack_invariants`
must succeed before `_build_source_terms` runs, because the source
builder uses unclamped/unfloored expressions whose validity DEPENDS on
the margin checks passing. If a margin check fires, the test
correctly aborts before the unclamped source builder produces a
mathematically-wrong residual.

The smoke test (`_independent_closure_smoke`) mirrors production
clamps/floors exactly, so it works regardless of margins — it's a pure
algebra-equivalence diagnostic.

---

## Section 2: Plan state after R6

`OwnedCoeffTracker` + 4-phase pipeline (static asserts → field-builders
→ runtime asserts → source-terms → independence → inject → solve) make
the implementation contract explicit. The plan is now structured so
that:

- Wrong ordering raises before the next phase can mask the error.
- Source independence is checked on the actual source UFL.
- Owned constants don't pollute across meshes or tests.
- Type-based filtering is gone; identity-based whitelist only.

(Full revised PLAN.md lands via auto-revise after the loop closes.)

---

## Section 3: Continued critique prompt

Loop now extended to cap = 7 (was 5). Two rounds remain after this one.

Review my responses to the 5 R5 issues. Push back where any "accept"
is superficial. Raise any new issues these structural changes
introduce. Re-issue anything from R1–R4 you don't think I addressed.
Same numbered format and verdict line:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
