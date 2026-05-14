# Correctness Audit — Chunk A

**Scope**: `solver_demo_slide15_no_speculative_cs.py`, `dispatch.py`, `mesh.py`, `nondim.py`, `_bv_common.py` (targeted reads)
**Auditor**: claude-sonnet-4-6
**Date**: 2026-05-13

---

## Q1: Stage 2 bump-loop — corrupt ctx_anchor["U"] after failed solve

**SEVERITY**: warning

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:454–498`

**TRIGGER**: A bump rung fails mid-sequence (e.g., bump to 0.50 fails after 0.20 succeeded).

**EVIDENCE**:
```python
for cs_target in bump_ladder:
    try:
        set_stern_capacitance_model(ctx_anchor, float(cs_target))
        with adj.stop_annotating():
            ctx_anchor["_last_solver"].solve()          # <- may throw mid-solve
        bump_history.append((float(cs_target), "ok"))
    except Exception as exc:
        bump_err = f"bump to C_S={cs_target} failed: ..."
        bump_history.append((float(cs_target), "fail"))
        break
if bump_err is not None:
    ...
    return { ... "method": ["stern-bump-failed"] * NV ... }
```

`ctx_anchor["U"]` **is corrupt** after a failed `.solve()`. Firedrake's SNES solver writes into `U` in place during its Newton iterations; a failed solve may leave `U` mid-iteration. However, the code **does correctly guard**: on `bump_err is not None` it immediately `return`s a failure record before ever touching `ctx_anchor` again for Stage 3. No corrupt U reaches Stage 3.

Minor issue: `set_stern_capacitance_model` has already mutated `ctx_anchor["nondim"]["bv_stern_capacitance_model"]` and `ctx_anchor["stern_coeff_const"]` to the failed rung value. If the caller somehow reused `ctx_anchor` after the early-return (which the current code does NOT do), it would see an incorrect C_S. As written, the ctx is discarded. **No bug in current code path.**

---

## Q2: Shared mutable state across K0_R4E_FACTORS loop

**SEVERITY**: note

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:740` (factor loop), `_make_sp` at lines 126–244

**TRIGGER**: Multiple factors run in sequence sharing the same `mesh` object.

**EVIDENCE**:
- `setup_firedrake_env()` only sets `os.environ.setdefault(...)` — idempotent, no global mutable state.
- `make_bv_solver_params` returns a new `SolverParams` (frozen dataclass) each call; `with_solver_options` returns another new frozen instance (`dataclasses.replace`). No sharing.
- The `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` / `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC` objects are imported from `_bv_common` — these are dict literals created at module import. They are passed into `make_bv_solver_params` which copies them into the new sp. No shared mutable dict.
- The `mesh` object IS shared across all factors. Firedrake meshes are immutable in coordinates once built; the ctx/forms/U created per factor each use their own FunctionSpace built on the shared mesh. There is no known Firedrake bug here, and sharing the mesh is the intended pattern.

**No cross-factor shared mutable state identified.**

---

## Q3: _stern_bump_ladder edge cases

**SEVERITY**: warning

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:304–322`

**TRIGGER**: `target == STERN_ANCHOR (0.10)` or `target < STERN_ANCHOR`.

**EVIDENCE** (verified by running the logic):
```
target=0.10  -> [0.10]   # "bump" to same value; solve() called at same C_S (wasteful but harmless)
target=0.05  -> [0.05]   # LOWERS C_S below anchor build value!
target=0.20  -> [0.20]   # correct single-step
target=0.30  -> [0.20, 0.30]  # correct two-step
target=100.0 -> full ladder  # correct
target=200.0 -> full ladder + 200.0  # correct (200 appended)
```

The `target <= STERN_ANCHOR` branch returns `[target]`, which for `target < STERN_ANCHOR` would call `set_stern_capacitance_model(ctx, target)` with a value **below** the anchor's build C_S. This violates the bump-ladder invariant and could destabilize the solve.

**In practice**, `main()` only passes `stern_final >= 0.20` (production) or `100.0` (no-stern). The `--stern-final` CLI arg has no lower-bound validation. A user passing `--stern-final 0.05` would silently lower C_S rather than error. Low priority since production paths are safe, but `_stern_bump_ladder` should raise on `target < STERN_ANCHOR` (not on the `==` case which is a harmless no-op).

---

## Q4: _grab callback — partial ctx, None/NaN handling

**SEVERITY**: note

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:520–538`

**TRIGGER**: Grid point fails warm_walk_phi.

**EVIDENCE**:
In `solve_grid_with_anchor` (grid_per_voltage.py:1107–1113), `per_point_callback` is called **only when `ok` is True** (converged). When `ok` is False, the callback is not called at all. So `_grab` is never invoked with a failed/partial ctx.

`cd_arr` and `pc_arr` are initialized to `np.nan`. For failed points, `_grab` is not called, so they remain `np.nan`. In `_to_json_list` (line 560–563):
```python
def _to_json_list(arr):
    return [
        float(x) if (np.isfinite(x) and converged[i]) else None
        for i, x in enumerate(arr)
    ]
```
NaN entries (failed points) become `None` in JSON. Correct.

**No bug.** The only gap: within `_grab` itself, if `_build_bv_observable_form` raises, `cd_arr[orig_idx]` stays NaN while `pc_arr[orig_idx]` might still get set (or vice versa). This creates a cd/pc asymmetry at a point where `converged[i]` is True, so the `_to_json_list` call would emit `None` for cd (nan) but a valid float for pc — which is internally inconsistent but not incorrect per the guard logic. Minor edge case, not a crash.

---

## Q5: sp.with_solver_options — in-place mutation vs. new object

**SEVERITY**: note — **NO BUG**

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:238–242`

**EVIDENCE**:
```python
sp = sp.with_solver_options(new_opts)
```
`SolverParams` is a `@dataclass(frozen=True)`. `with_solver_options` uses `dataclasses.replace(self, solver_options=opts)` — returns a **new** frozen instance. The local `sp` is rebound; the old object is unreferenced. `sp_baseline` and `sp_anchor_cs` are built in separate `_make_sp` calls and receive separate patches. No shared convergence dict.

---

## Q6: dispatch.py — unknown formulation behavior

**SEVERITY**: note

**LOCATION**: `Forward/bv_solver/dispatch.py:68–79`

**EVIDENCE**:
```python
def _resolve_backend(solver_params):
    formulation = _read_formulation(solver_params)
    if formulation == "logc_muh":
        return "logc_muh"
    return "logc"  # default fallback
```
Unknown formulations (e.g., a typo like `"logc_mu"`) silently fall through to `"logc"`. The docstring says `_validate_formulation` in config.py rejects bad names at parse time. If that guard is bypassed (e.g., in tests constructing raw dicts), dispatch silently uses the wrong backend with no error. **Not a crash, but wrong results silently.** Defensive behavior is intentional per the docstring — noted for awareness.

Also: `_read_bv_convergence_field` reads `solver_params[10]` to get `params_dict`. If `solver_params` is a `SolverParams` (11-element), `params[10]` is `solver_options` dict. But if `solver_params` is a legacy 11-tuple that stores `bv_convergence` inside `solver_options`, this chain works. If someone passes a shorter tuple, the `except Exception: params_dict = None` swallows the IndexError and returns the default — silently misrouted. Not a production concern given current callers.

---

## Q7: mesh.py make_graded_rectangle_mesh — grading sum and electrode marker

**SEVERITY**: note — **NO BUG**

**LOCATION**: `Forward/bv_solver/mesh.py:102–108`

**EVIDENCE** (verified numerically):
- `RectangleMesh(8, 80, 1.0, 1.0)` creates a unit square; y-coords are k/80 for k=0..80.
- Stretch: `y -> (y^3.0) * 1.0` (domain_height_hat = L_EFF_M/1e-4 = 100e-6/1e-4 = 1.0).
- Sum of y-increments = 1.0 exactly (no roundoff at this scale — verified: `abs(sum - 1.0) < 1e-14`).
- `y[0] = 0.0` is the electrode (marker 3 = bottom per Firedrake RectangleMesh convention). Correct per docstring.
- Grading ratio max/min = 18961 — very aggressive near y=0, which is the intent (EDL resolution).

No issue. Note: the mesh is built with `domain_height_hat = L_EFF_M / 1.0e-4` (line 732 of demo). When `L_EFF_M = 100e-6 m`, `domain_height_hat = 1.0` — correct. If `L_EFF_M` were changed, the denominator `1.0e-4` is a hardcoded L_REF assumption. This is correct for the current codebase where `L_REF = 100 µm`, but fragile if L_REF is ever changed without updating this line.

---

## Q8: _factor_label with {:g} formatting

**SEVERITY**: note — **NO BUG**

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:247–248`

**EVIDENCE** (verified):
```python
factors = [1.0, 1e-6, 1e-12, 1e-18]
labels  = ['factor_1', 'factor_1e-06', 'factor_1e-12', 'factor_1e-18']
```
All four are distinct. `{:g}` drops trailing zeros and switches to scientific notation at small magnitudes — exactly the right format here. No collision.

---

## Q9: adj.stop_annotating() — exception lifecycle

**SEVERITY**: warning

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:397–409` (Stage 1), `457–458` (Stage 2)

**TRIGGER**: An unexpected exception propagates out of the `with adj.stop_annotating():` block.

**EVIDENCE**:
Stage 1:
```python
try:
    with adj.stop_annotating():
        anchor_result = solve_anchor_with_continuation(...)
    anchor_converged = True
except LadderExhausted as exc:
    anchor_err = ...
except Exception as exc:
    anchor_err = ...
```
The `with` statement guarantees `__exit__` is called even on exception — Python's context manager protocol ensures annotation is re-enabled when the block exits by exception. So `stop_annotating()` is correctly unwound for caught exceptions.

Stage 2 bump loop:
```python
with adj.stop_annotating():
    ctx_anchor["_last_solver"].solve()
```
Each rung uses its own `with` block — correctly scoped.

**One subtle concern**: `solve_grid_with_anchor` internally wraps its entire grid loop in `with adj.stop_annotating():` (grid_per_voltage.py:1060). This is called from Stage 3 **without** an outer `adj.stop_annotating()` in the demo. If an unhandled exception propagates out of `solve_grid_with_anchor` during Stage 3, the context manager in that function exits cleanly. No lifecycle issue.

**No bug found**, but if `firedrake.adjoint.stop_annotating()` uses a non-reentrant counter (depth-based rather than flag-based), nested calls (Stage 1's outer `with` wrapping an inner one inside `solve_anchor_with_continuation`) could be problematic. This is a Firedrake internal; from the usage pattern here (each `with` has a matching exit), it is not an issue as long as Firedrake implements it as a counter (which is the standard pyadjoint pattern).

---

## Q10: Additional findings

### 10a: domain_height_hat hardcoded L_REF assumption

**SEVERITY**: warning

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:732`

```python
domain_height_hat = L_EFF_M / 1.0e-4
```

The denominator `1.0e-4` is L_REF = 100 µm, hardcoded as a magic number. If `L_EFF_M` or L_REF changes, this silently produces wrong domain height. Should reference a `L_REF` constant from `_bv_common` or use `C_SCALE`-aligned nondim. Currently safe since L_REF is fixed at 100 µm throughout the codebase.

### 10b: _to_json_list gates on converged[i] but cd/pc arrays may be inconsistent

**SEVERITY**: note

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:560–563`

```python
def _to_json_list(arr):
    return [
        float(x) if (np.isfinite(x) and converged[i]) else None
        for i, x in enumerate(arr)
    ]
```

If a converged point's `_grab` call partially succeeds (cd captured, pc raises — or vice versa), the resulting JSON will show `None` for the failed observable and a float for the successful one at the same voltage index. This is inconsistent but won't cause a crash. The `and converged[i]` gate means unconverged NaN always -> None, which is correct.

### 10c: `sp_baseline` passed to Stage 3 has C_S = stern_final, not STERN_ANCHOR

**SEVERITY**: note — intentional but worth flagging

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:375–379`, `543`

`sp_baseline` is built with `stern_capacitance_f_m2=stern_final_v` (0.20) and passed to `solve_grid_with_anchor`. Inside `_build_for_voltage`, a fresh ctx is built with `sp_baseline` — so each grid point builds its forms with C_S=0.20 baked in. The Stage 2 bump that mutated `ctx_anchor["stern_coeff_const"]` to 0.20 is then consistent with what the grid solve builds. This is correct by design. However, the anchor's live `stern_coeff_const` is NOT forwarded to Stage 3 ctx objects — each Stage 3 ctx builds its own Stern constant from `sp_baseline`'s config. This means the Stage 2 bump's runtime mutation of `ctx_anchor` does not matter for Stage 3 correctness (Stage 3 gets C_S=0.20 from sp_baseline regardless). The bump's purpose is to produce the correct `U_post_bump` snapshot which IS used for Stage 3 initialization. Correct.

### 10d: No validation that anchor_v_rhe is in V_RHE_GRID

**SEVERITY**: note

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:385`, `540`

`ANCHOR_V_RHE = +0.55` is the last element of `V_RHE_GRID = linspace(-0.40, +0.55, 25)`. The code passes `phi_grid_eta` (all 25 points including the anchor voltage) to `solve_grid_with_anchor`. The anchor voltage will be the nearest source for itself — so it warm-walks from V=+0.55 to V=+0.55 (zero-step walk). This is harmless (the walk succeeds trivially) but a small wasted solve. If the user passes `--anchor-voltage` to a value not in the grid, the anchor is correctly used as the external source but the anchor voltage itself won't appear in the output arrays (not a bug). The `solve_grid_with_anchor` docstring does not require the anchor to be in the grid.

### 10e: summary.json uses _config_dict(0.0, ...) — factor hardcoded to 0.0

**SEVERITY**: note

**LOCATION**: `scripts/studies/solver_demo_slide15_no_speculative_cs.py:769–772`

```python
"config": _config_dict(
    0.0, no_stern=no_stern, anchor_v_rhe=anchor_v_rhe,
    initializer=initializer, stern_final=stern_final,
),
```

The summary-level config embeds `"K0_R4e_factor": 0.0` which is not a real factor value. This is a documentation artifact (per-factor data is in per_factor subdirs), but could confuse downstream readers of `summary.json` who expect `K0_R4e_factor` to be meaningful. Low impact; purely cosmetic.

---

## VERDICT: CONCERNS FOUND

**Critical bugs**: None.

**Warnings** (should be reviewed/fixed):
1. **Q3**: `_stern_bump_ladder(target < STERN_ANCHOR)` silently lowers C_S rather than raising. The `--stern-final` CLI arg has no lower-bound validation. A user passing `--stern-final 0.05` would break the bump assumption without any error.
2. **Q9**: Nested `adj.stop_annotating()` (demo outer + `solve_anchor_with_continuation` inner) is only safe if pyadjoint uses a reference-count approach — which it does in standard pyadjoint, but this is an implicit assumption on a Firedrake internal. Worth a comment.
3. **Q10a**: `domain_height_hat = L_EFF_M / 1.0e-4` hardcodes L_REF as a magic number; would silently break if L_EFF_M or L_REF changes.

**Notes** (low priority):
- Q4: cd/pc asymmetry possible if `_grab` partially fails at a converged point.
- Q6: dispatch silently falls back to `"logc"` for unknown formulations.
- Q10b/c/d/e: minor cosmetic/documentation issues; no functional impact.
