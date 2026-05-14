# Correctness Audit — Chunk C (Sonnet)

Files audited:
- `Forward/bv_solver/anchor_continuation.py` (targeted regions)
- `Forward/bv_solver/grid_per_voltage.py` (targeted regions)
- `Forward/bv_solver/observables.py` (full)

Date: 2026-05-13

---

## FINDINGS

---

### Q1 — AdaptiveLadder midpoint insertion: prev=0 / FP collapse

**SEVERITY: LOW (warm_start_floor path) / CLEAR (geometric path)**

**Evidence:**

Geometric path (post-first-rung):
```python
midpoint = math.sqrt(prev * scale)
self._planned.insert(self._idx, float(midpoint))
```
No guard against `midpoint == prev` or `midpoint == scale` due to FP. Concretely: `prev=1e-300, scale=1e-300` → `midpoint = 1e-300` — insertion of an identical value, causing an infinite loop (failure always triggers the same midpoint, which never advances `_idx`). This is an edge case only reachable if the orchestrator builds a ladder with two numerically-equal adjacent entries; the production `initial_scales = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)` has no such pair. But it IS reachable after several recursive inserts that collapse the gap below `sqrt`'s precision floor. The `max_inserts_per_step=4` cap prevents an actual infinite loop; after 4 identical inserts `record_failure_and_insert` returns `False` and the orchestrator raises `LadderExhausted`. **Not a true infinite loop, but wastes the per-step budget on do-nothing retries.**

Warm_start_floor path (first-rung failures):
```python
midpoint = 0.5 * (self._warm_start_floor + scale)
if not (self._warm_start_floor < midpoint < scale):
    return False
```
Explicit guard present. `sqrt(0)=0` scenario is **not reached** here because first-rung failures use arithmetic midpoint, not geometric. **CLEAR.**

**Verdict on Q1: LOW. Geometric path has no FP-collapse guard; mitigated by the insert cap, but wastes budget. No actual infinite loop exists.**

---

### Q2 — `_inserts_at_current_step` reset semantics after cap

**SEVERITY: CLEAR**

**Evidence:**
```python
def record_failure_and_insert(self) -> bool:
    ...
    if self._inserts_at_current_step >= self._max_inserts_per_step:
        return False   # <- cap check is FIRST, before any insert
    ...
    self._inserts_at_current_step += 1
    return True

def record_success(self) -> None:
    ...
    self._idx += 1
    self._inserts_at_current_step = 0   # <- reset only on success
```

After `max_inserts_per_step` insertions at one rung, the next call to `record_failure_and_insert` hits the cap check immediately and returns `False`. The caller (`_run_k0_ladder`) sees `False` and returns `(False, k0_lad, k0_last_ok_snap)`. The outer code raises `LadderExhausted`. No silent advance — the path is correct. **CLEAR.**

---

### Q3 — `is_done()` semantics: off-by-one risk

**SEVERITY: CLEAR**

**Evidence:**
```python
def is_done(self) -> bool:
    return self._idx >= len(self._planned)

def record_success(self) -> None:
    ...
    self._idx += 1          # advances PAST the last rung
    self._inserts_at_current_step = 0
```

After the last rung (`self._planned[-1] == 1.0`) is recorded as success, `_idx` becomes `len(self._planned)` and `is_done()` returns `True`. The outer `while not k0_lad.is_done()` loop exits. The orchestrator then checks `converged_to_target = ok`. This is set correctly inside `_run_k0_ladder` — `ok` is `True` after the last `run_ss`. **No off-by-one. CLEAR.**

---

### Q4 — `solve_anchor_with_continuation` outer loop: max-iter safety

**SEVERITY: LOW**

**Evidence:**

The `_run_k0_ladder` helper wraps a `while not k0_lad.is_done()` loop. `AdaptiveLadder` is bounded by `max_inserts_per_step` at every rung. Since the planned list grows by at most `max_inserts_per_step` per rung and each rung either succeeds (advancing `_idx`) or exhausts inserts (returning `False`), the worst-case iterations = `len(initial_scales) * (max_inserts_per_step + 1)`. Production: `5 * 5 = 25` iterations max. There is no unbounded outer loop. **CLEAR.**

---

### Q5 — `NonlinearVariationalSolver` solver_parameters defaults

**SEVERITY: MEDIUM**

**Evidence:**
```python
params_block = sp[10] if hasattr(sp, "__getitem__") else {}
items = (
    params_block.items() if isinstance(params_block, dict) else []
)
solve_opts = {k: v for k, v in items if k not in NON_PETSC_KEYS}
solve_opts.setdefault("snes_error_if_not_converged", True)
```

Only `snes_error_if_not_converged` is defaulted. If `sp[10]` is an empty dict or missing `ksp_type`, `pc_type`, `snes_type`, PETSc falls back to its compiled-in defaults (typically `snes_type=newtonls`, `ksp_type=gmres`, `pc_type=ilu`). For this solver the expected configuration is `snes_type=newtonls + ksp_type=preonly + pc_type=lu` (direct). Without `ksp_type=preonly + pc_type=lu`, iterative GMRES+ILU degrades or diverges on highly ill-conditioned PNP Jacobians at high Debye-layer refinement. **This is a usage-time bug only** — it manifests when a caller passes an incomplete `sp[10]`. Production drivers (`scripts/_bv_common.py`) presumably set full PETSc options; but no validation is done here.

**TRIGGER:** Any caller that passes `sp[10] = {}` or `sp[10] = {"snes_error_if_not_converged": True}` will silently get GMRES+ILU instead of LU.

**VERDICT: MEDIUM — no validation of required PETSc keys; silent wrong solver choice possible.**

---

### Q6 — `ctx['_last_solver']` reuse across Stage 1 → Stage 3: CRITICAL PATH

**SEVERITY: CLEAR (correctly isolated)**

**Evidence:**

Stage 1 (`solve_anchor_with_continuation`) builds its own ctx and stores `ctx["_last_solver"] = solver`. This ctx is returned in `AnchorContinuationResult.ctx`.

Stage 3 (`solve_grid_with_anchor`) calls `_build_for_voltage(target_phi)` for **every grid point**, which:
```python
def _build_for_voltage(phi_target: float):
    sp_v = _params_with_phi(phi_target)
    ctx = build_context(sp_v, mesh=mesh)           # FRESH ctx
    ctx = build_forms(ctx, sp_v)                   # FRESH forms
    ...
    solver = fd.NonlinearVariationalSolver(problem, ...)   # FRESH solver
    ctx["_last_solver"] = solver
    of_cd = _build_bv_observable_form(ctx, ...)    # FRESH observable
    return ctx, solver, of_cd
```

Stage 3 **never touches** Stage 1's ctx or `_last_solver`. The Stern bump (Stage 2 demo loop) that mutates Stage 1's ctx via `stern_coeff.assign(...)` is completely isolated from Stage 3's freshly-built contexts. **No cross-stage solver/form aliasing. CLEAR.**

---

### Q7 — `set_stern_capacitance_model` nondim conversion

**SEVERITY: CLEAR**

**Evidence:**
```python
factor = float(scaling.get("bv_stern_phys_to_nondim_factor", 1.0))
nondim_value = float(c_s_f_m2) * factor
stern_const.assign(nondim_value)
```

The factor defaults to `1.0` when missing — this would silently assign physical F/m² directly as the nondimensional constant, **skipping the conversion**. However, `bv_stern_phys_to_nondim_factor` is stashed at form-build time by the form builder whenever `stern_capacitance_f_m2 > 0` is set, so in practice the key is always present on a Stern-enabled ctx. The missing-key fallback of `1.0` is a silent wrong-value bug for any hypothetical ctx that lacks it.

**SEVERITY: LOW** — Reachable only if ctx is manually constructed without the factor (not a production path). The inverse `get_stern_capacitance_model` also uses `factor` with a zero-guard, consistent.

---

### Q8 — `snapshot_U` / `restore_U` atomicity

**SEVERITY: CLEAR**

**Evidence:**
```python
def _snapshot_U(U) -> tuple:
    return tuple(d.data_ro.copy() for d in U.dat)

def _restore_U(snap: tuple, U, U_prev) -> None:
    for src, dst in zip(snap, U.dat):
        dst.data[:] = src
    U_prev.assign(U)       # <- U_prev is updated AFTER U is written
```

`restore_U` writes `U.dat` first, then does `U_prev.assign(U)`. This means U_prev is always assigned from the just-restored U, so they agree. SER's first step after restore computes the time-residual using `U - U_prev` = 0, i.e. the transient term vanishes at the first step. This is correct: a restored state is effectively "U_prev = U at the restore point" which prevents a spurious transient kick. **CLEAR.**

---

### Q9 — `warm_walk_phi._march` recursion: paf state after failure

**SEVERITY: MEDIUM**

**Evidence:**

At max bisect depth, the failure path is:
```python
if depth >= bisect_depth:
    _restore_U(ckpt_outer, U, U_prev)
    paf.assign(float(v0))
    return False
```
`paf` is reset to `v0` at the leaf. The caller's `_march` call:
```python
if not _march(v_prev_substep, v_mid, depth + 1):
    _restore_U(ckpt_outer, U, U_prev)
    paf.assign(float(v0))   # caller also resets paf to its own v0
    return False
```
Each level resets paf to its own `v0`, which is the correct previous-successful-V for that recursion level. The chain unwinds correctly.

**However, there is a subtle issue with the final SS at `v_target_eta`:**

```python
if not _march(float(v_anchor_eta), float(v_target_eta), 0):
    return False    # paf was reset to v_anchor_eta by _march's failure path
# Final SS at V_target to make sure we're firmly at the target.
paf.assign(float(v_target_eta))
if not run_ss(max_ss_steps_final):
    return False    # <- paf is LEFT AT v_target_eta; U may be partially converged
return True
```

On final-SS failure, `warm_walk_phi` returns `False` but **does NOT restore U to the entry snapshot**. The caller (`solve_grid_with_anchor`) sees `ok=False` and marks the point as failed, but `U` is NOT rolled back to the anchor state and `paf` is stuck at `v_target_eta`. Since Stage 3 builds a **fresh ctx per grid point**, this stale U/paf affects no subsequent point. **CLEAR in Stage 3.**

But in any context that **reuses** the ctx after a `warm_walk_phi` call (e.g. if `warm_walk_phi` is called standalone), the stale U/paf is a latent bug. The docstring says "On failure, restores the U snapshot taken at function entry (the anchor state) and resets `ctx['phi_applied_func']` to `v_anchor_eta`" — this claim is **incorrect for final-SS failure**. Only `_march` failure triggers the entry-snapshot restore; `run_ss` failure at the end does not.

**SEVERITY: MEDIUM — docstring makes a false promise. Stale U+paf after final-SS failure. Benign in production Stage 3 (fresh ctx per point), but a hazard for any reuse caller.**

---

### Q10 — `make_run_ss` plateau detection: false plateau on sign flip

**SEVERITY: LOW**

**Evidence:**
```python
delta = abs(fv - prev_flux)
sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)
```

If `fv` oscillates between `+ε` and `-ε` (sign flip near zero), then `delta = 2ε` and `sv = max(ε, ε, ss_abs_tol) = ss_abs_tol` (if ε < ss_abs_tol). Then `delta / sv = 2ε / ss_abs_tol` — this is `<= ss_rel_tol` only if `2ε <= ss_rel_tol * ss_abs_tol = 1e-4 * 1e-8 = 1e-12`. So at physical currents this threshold is rarely triggered. The `(delta <= ss_abs_tol)` branch fires when `2ε <= 1e-8` — essentially zero current. A real sign-flip oscillation during transients would have `delta` much larger than `ss_abs_tol`. **CLEAR in practice, but the OR-short-circuit means a near-zero constant oscillation could produce false-positive steadiness.** Low-severity edge case.

---

### Q11 — `make_run_ss` dt growth: unbounded growth risk

**SEVERITY: CLEAR**

**Evidence:**
```python
if prev_delta is not None and delta > 0:
    ratio = prev_delta / delta
    if ratio > 1.0:
        grow = min(ratio, dt_growth_cap)          # capped at dt_growth_cap
        dt_val = min(dt_val * grow, dt_max)       # capped at dt_max
    else:
        dt_val = max(dt_val * 0.5, float(dt_init))
    dt_const.assign(dt_val)
```

Growth is double-capped: per-step by `dt_growth_cap` and globally by `dt_max = dt_init * dt_max_ratio`. Production defaults: `dt_growth_cap=4.0`, `dt_max_ratio=20.0`, so `dt_max = 0.25 * 20 = 5.0`. **Cannot grow unboundedly. CLEAR.**

---

### Q12 — `U_prev` update in SER: ordering

**SEVERITY: CLEAR**

**Evidence:**
```python
for _ in range(1, max_steps + 1):
    try:
        solver.solve()
    except Exception:
        return False
    U_prev.assign(U)     # <- assigned AFTER each accepted solve
    fv = float(fd.assemble(of_cd))
    ...
```

`U_prev.assign(U)` happens unconditionally after every successful Newton solve, before the next iteration. The SER time-stepping residual `(U - U_prev) / dt` sees the correct previous state each step. **CLEAR.**

---

### Q13 — `solve_grid_with_anchor` source pool: "nearest to target" semantics

**SEVERITY: CLEAR**

**Evidence:**
```python
src_phi, src_snap = min(
    sources,
    key=lambda s: abs(float(s[0]) - target_phi),   # <- distance to TARGET
)
```

The key is `|src_phi - target_phi|`, not `|src_phi - anchor_phi|`. As `sources` grows with each successful grid point, the source is always the nearest **already-converged** voltage to the current target. This is the intended nearest-neighbor warm-start. **CLEAR.**

---

### Q14 — `per_point_callback` on failure

**SEVERITY: LOW (design choice, not a bug)**

**Evidence:**
```python
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

Callback is **only called on success**. On failure, `cd_arr[i]` / `pc_arr[i]` stay at whatever the caller's initializer set them to (typically `np.nan`). This is a consistent design — callers that index by `converged_indices()` won't see stale data. But **callers that iterate over all points and don't check `converged` could silently read nan**. Not a bug in this file; potential misuse downstream. **LOW / informational.**

---

### Q15 — `observables.py` current sign convention

**SEVERITY: CLEAR**

**Evidence:**
```python
return scale_const * rate_sum * ds(electrode_marker)
```

`R_j` is the BV rate expression from `ctx["bv_rate_exprs"]`. For cathodic ORR at overpotential η < 0, the Butler-Volmer law gives a positive reduction rate (O₂ consumed, current flowing into electrode). The sign of the assembled observable depends on the **calling convention**: the demo uses `scale=-I_SCALE` to flip the sign to the cathodic-negative convention used for RRDE plots. There is no sign error internal to `observables.py`; the sign is entirely deferred to `scale`. The `peroxide_current = R_0 - R_1` form is noted as DEPRECATED with the correct reason. **CLEAR.**

---

### Q16 — Additional findings

#### Q16.A — `_run_gamma_picard` first-iteration rollback missing

**SEVERITY: MEDIUM**

**Evidence** (lines 1148–1168):
```python
for _it in range(picard_max_iters):
    gamma_old = float(ctx["cation_hydrolysis"].gamma_func)
    gamma_new = update_gamma_from_solution(ctx)
    gamma_history.append(float(gamma_new))
    lam_active = float(ctx["cation_hydrolysis"].lambda_hydrolysis_func)
    if lam_active == 0.0:
        break
    rel = abs(gamma_new - gamma_old) / max(abs(gamma_new), abs(gamma_old), 1e-30)
    if rel < picard_rel_tol:
        break
    ok = run_ss(max_ss_steps_per_rung)
    if not ok:
        break   # <- ACCEPTS partial Picard state without rollback
```

When `run_ss` fails mid-Picard, the function breaks and returns silently with no rollback. The ctx is left at a **Newton-diverged U** (the failed `solver.solve()` inside `run_ss` raised, returning `False`, but the FE state is indeterminate — Firedrake's Newton may have partially updated U before diverging). The outer orchestrator receives a `gamma_picard_history` but no indication of Picard failure; `converged_to_target` is **not affected** because the Picard runs **after** the ladder has already set it. This means a Picard-de-stabilized state is silently accepted as "converged" when Picard is called as a post-convergence step (`_run_gamma_picard("bare")`).

**TRIGGER:** `kw_eff_ladder=None` + `cation_hydrolysis` active + Picard de-stabilizes at the first iteration. Outcome: partial U reported as converged.

**FIX:** Capture a snapshot before the Picard loop; restore on `ok=False`; propagate a failure signal to the caller.

---

#### Q16.B — Lambda ladder: `lam_scales[-1] != 1.0` check is fragile for floating-point

**SEVERITY: LOW**

**Evidence** (line 1376):
```python
lam_scales = tuple(v / lam_target for v in lam_positive)
if not lam_scales or lam_scales[-1] != 1.0:
    raise ValueError(...)
```

`lam_target = float(lam_seq[-1])`. `lam_positive[-1] == lam_target` (same float). So `lam_positive[-1] / lam_target = 1.0` exactly when both are the same float object. In practice this is fine because both come from the same `lam_seq[-1]` conversion chain. But if `lam_target` were computed differently (e.g. from a rounded intermediate), exact `!= 1.0` would fire spuriously. **LOW / theoretical.**

---

#### Q16.C — `set_reaction_k0_model` aliasing guard is ineffective for Picard

**SEVERITY: LOW**

**Evidence** (lines 304–308):
```python
new_rxn = {**rxns[j], "k0_model": float(k0_model_value)}
new_rxns = list(rxns)
new_rxns[j] = new_rxn
nondim["bv_reactions"] = new_rxns
ctx["nondim"] = nondim
```

The comment says this "guards against aliasing if the dict was captured elsewhere." However, `nondim` is obtained via `ctx.get("nondim", {})` which returns the **same dict object** that `ctx["nondim"]` points to. Then `ctx["nondim"] = nondim` reassigns ctx to the same (now mutated) object. The `new_rxns = list(rxns)` copy protects the reactions list, but the outer `nondim` dict is mutated in place. Any caller that held a reference to `ctx["nondim"]` before the call will see the updated `"bv_reactions"` key — exactly the aliasing the comment claims to avoid. **LOW — the copy is shallow; the intent is partially achieved but the aliasing concern in the comment is misleading.**

---

#### Q16.D — `c_s_ladder` rollback on failure restores `cs_last_success_snap` not `last_success_snap`

**SEVERITY: LOW / BENIGN**

**Evidence** (lines 1299–1314):
```python
if not ok:
    if cs_last_success_snap is not None:
        restore_U(cs_last_success_snap, ctx["U"], ctx["U_prev"])
    ...
    raise LadderExhausted(...)
cs_last_success_snap = snapshot_U(ctx["U"])
last_success_snap = cs_last_success_snap
```

`cs_last_success_snap` and `last_success_snap` are kept in sync: after every C_S success, both are updated. After failure, restore uses `cs_last_success_snap`. This is correct. The only subtle case is the **first C_S rung failure before any success**: `cs_last_success_snap = last_success_snap` (initialized at line 1270), which is the initial k0-ramp IC snapshot. Restoring that is correct (rolls back to the pre-C_S state). **CLEAR.**

---

## VERDICT

**CRITICAL: 0**
No cross-stage solver aliasing, no actual infinite loops, no silent wrong-state acceptance that would corrupt results in the production Stage 1→2→3 pipeline.

**MEDIUM: 3**

1. **Q9 / `warm_walk_phi` final-SS failure does not restore U+paf** — docstring promise is false; benign in Stage 3 (fresh ctx) but a latent hazard for any caller that reuses ctx after failure.

2. **Q5 / missing PETSc key validation** — if `sp[10]` is missing `ksp_type`/`pc_type`, the solver silently falls back to GMRES+ILU which will likely diverge on PNP Jacobians. No assertion or warning.

3. **Q16.A / `_run_gamma_picard` Newton-diverged state silently accepted as converged** — when `run_ss` fails mid-Picard, U is left in an indeterminate Newton-partially-updated state but the outer orchestrator marks the run as converged (Picard failure is not propagated). Requires `cation_hydrolysis` active and Picard de-stabilization to trigger.

**LOW: 5**

- Q1: geometric midpoint FP collapse wastes insert budget (capped, not infinite).
- Q7: `bv_stern_phys_to_nondim_factor` missing-key fallback to 1.0 silently skips nondim conversion.
- Q10: OR-plateau detection could false-positive on near-zero sign-flip oscillation.
- Q16.B: `lam_scales[-1] != 1.0` exact-float comparison fragile.
- Q16.C: `set_reaction_k0_model` aliasing comment overstates protection.

**INFORMATIONAL: 1**

- Q14: `per_point_callback` not called on failure; callers that don't gate on `converged` will read nan.
