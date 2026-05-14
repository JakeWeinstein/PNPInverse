# Opus L3 Correctness Audit — Deep Algorithmic Reasoning

Scope: anchor_continuation.py + grid_per_voltage.py + observables.py.
Focus: ladder termination, warm-walk recursion, paf state, Stern bump
correctness, observable scale arithmetic, NaN propagation.

---

## A. AdaptiveLadder termination

LOCATION: `anchor_continuation.py:802-895`

**TERMINATION: GUARANTEED IN PRACTICE — but NOT by `max_inserts_per_step` count alone.**

The cap `max_inserts_per_step=4` is **per-`_idx` position**, not per
original rung. Mechanism:

- `record_failure_and_insert` increments `_inserts_at_current_step`
  and inserts a geometric midpoint at `_idx`.
- After insert: `_planned[_idx]` is the midpoint. Next attempt may
  succeed.
- `record_success` advances `_idx` AND **resets**
  `_inserts_at_current_step = 0` (line 836).
- The advanced `_idx` now points at the ORIGINAL failing rung. If
  that fails again, a fresh budget of 4 inserts is allowed.

This means: every level of successful bisection grants a fresh
insert budget at the next attempt. Total inserts at a given
"original gap" can theoretically exceed 4 — the structure is
recursive geometric bisection.

**However**, termination IS guaranteed because:

1. Each geometric midpoint `sqrt(prev*scale)` halves the **log-gap**.
2. After ~53 levels of halving, the log-gap falls below `float64`
   precision and `midpoint == prev` numerically. The guard at
   `record_failure_and_insert` for the geometric path does NOT
   explicitly check `midpoint > prev` — it inserts blindly via
   `_planned.insert(_idx, float(midpoint))`. With `prev == midpoint`,
   the new midpoint equals `prev`, so the **next call to
   `current_scale` returns `prev`**.
3. If that `prev`-equal scale fails (the SS solver should also
   converge — but say it fails), 4 inserts produce 4 more
   `prev`-equal midpoints. They each retry. They all fail (the state
   doesn't change). `_inserts_at_current_step = 4 >= 4` → `False` →
   `LadderExhausted`.

So termination falls out of the per-position cap once the gap
collapses, but only AFTER float underflow stops bisection from
producing distinguishable midpoints. **The arithmetic-bisection
branch (warm_start_floor)** at line 879 DOES check `floor < mid <
scale` — so it terminates explicitly on numerical-floor collapse.
The geometric branch lacks that guard.

SEVERITY: LOW — does not cause infinite loops in normal use, only
in pathological "Newton converges nowhere" scenarios. The PETSc
backend's `snes_error_if_not_converged: True` propagates exceptions,
not silent stuck loops.

EVIDENCE: Line 884-886 has no `prev < midpoint < scale` guard for
geometric inserts (unlike the warm_start_floor branch at 879).

VERDICT: Not a bug; defensible defensive guard would harden the
geometric branch symmetrically.

---

## B. warm_walk_phi._march recursion correctness

LOCATION: `grid_per_voltage.py:271-303`

**v_prev_substep tracking: CORRECT.**

Traced scenario:

1. `_march(anchor, target, depth=0)`, n_substeps=8.
   `substeps = linspace(anchor, target, 9)[1:]` (8 points).
   `v_prev_substep = anchor`.
2. Substeps 1-4 succeed: each advances `v_prev_substep` to v_sub
   (line 285).
3. Substep 5 (v_5) fails:
   - `ckpt_inner` rolled back (line 287); `paf.assign(v_4)` (line 288).
   - depth=0 < bisect_depth=5; v_mid = 0.5*(v_4 + v_5).
   - Recursive `_march(v_4, v_mid, depth=1)`:
     - **n_substeps=8 INSIDE recursion (closure captures outer
       n_substeps).** Confirmed at line 278: `np.linspace(v0, v1,
       n_substeps + 1)[1:]` — the `n_substeps` is from the closure,
       not a parameter. So each recursion subdivides
       `[v_prev_substep, v_sub]` into 8 substeps again — not halved.
   - Both recursions succeed.
   - Line 302: `v_prev_substep = float(v_sub) = v_5`.
4. Substep 6 fails:
   - `v_prev_substep` is correctly v_5 (line 302's update).
   - `paf.assign(v_5)` (line 288 with the new substep's
     v_prev_substep).
   - `v_mid' = 0.5*(v_5 + v_6)`.
   - Recursion `_march(v_5, v_mid', depth=1)`.

**The CRITICAL DESIGN NOTE in the comment at line 272-277 is
correct**: tracking `v_prev_substep` separately from `paf` (which is
NOT in U.dat and so does NOT roll back on `_restore_U`) is what
makes this bisection correct.

**One subtle behavior**: when both recursions succeed, U is at the
state corresponding to `v_sub`. Outer loop proceeds to substep+1
without re-running SS at v_sub — that's fine because the inner
recursion's final substep IS the SS at v_sub. **CORRECT.**

SEVERITY: PASS — no bug.

**Subtle observation**: each recursive depth uses **n_substeps=8
fresh substeps over a 1/8-sized interval**, so the effective grid
resolution at depth=d is `1/8^(d+1)`. At bisect_depth=5, max
resolution ~`1/8^6 ≈ 4e-6` of the original gap. This is overkill
unless V_T-scale features are present. Cost: 8 substeps × 8^d ≈
exponential in depth. The depth cap controls cost.

VERDICT: CORRECT.

---

## C. paf state across grid points

LOCATION: `forms_logc_muh.py:375-376`,
`grid_per_voltage.py:1010-1087`

**paf initialization in fresh ctx: BAKED IN AT BUILD TIME, then
explicitly OVERWRITTEN by the orchestrator.**

Sequence at line 1010-1036 (`_build_for_voltage`):

1. `sp_v = _params_with_phi(phi_target)` — `sp.phi_applied` is the
   TARGET grid voltage.
2. `build_context(sp_v, mesh)` → `build_forms(sp_v)` →
   `forms_logc_muh.py:376` does `phi_applied_func.assign(
   scaling["phi_applied_model"])` — the TARGET voltage.

So freshly built ctx has paf == target. Then at line 1085-1087:

3. `_restore_U(src_snap, U, U_prev)` — restore U from neighbour.
4. `_set_z_factor(ctx, 1.0)` — full z.
5. `ctx["phi_applied_func"].assign(float(src_phi))` — OVERWRITE
   paf to **source** voltage, so the residual sees source-V at the
   first Newton solve, matching the restored U.

**This is CORRECT** — paf is overwritten BEFORE `warm_walk_phi` is
called. So the residual sees a self-consistent (src_V, U_src) pair
at the start of the walk.

**Potential mini-issue**: the IC routine (`set_initial_conditions`
at line 1018, between k0 pin and re-pin) runs at paf == target_V
(the fresh build value), but the IC seeds U from scratch — that U
is immediately discarded by `_restore_U` at line 1085. So this is
**harmless dead work** but a (minor) inefficiency: the IC's V-state
is irrelevant to the walk, and the IC could be skipped entirely on
warm-walk grid points. Not a correctness issue.

SEVERITY: PASS (minor wasted work, no bug).

VERDICT: CORRECT.

---

## D. extract_preconverged_anchor timing (POST-bump)

LOCATION: `anchor_continuation.py:191-239`,
`solver_demo_slide15...py:454-514`

**POST-bump snapshot: CORRECT in the slide15 demo.**

In `solve_anchor_with_continuation`, `result.U_data =
last_success_snap if converged_to_target else None` (line 1611).
`last_success_snap` is updated after every successful k0 rung
(line 1221).

But the slide15 demo runs C_S bump OUTSIDE
`solve_anchor_with_continuation`:

1. Stage 1: `solve_anchor_with_continuation(...,
   stern_capacitance_f_m2=STERN_ANCHOR)` returns result with U_data
   at C_S=STERN_ANCHOR.
2. Stage 2: bump loop at line 454-463 mutates the Stern Constant on
   `ctx_anchor` and re-solves via `ctx_anchor["_last_solver"].solve()`.
   After each bump, U has converged at the new C_S.
3. Line 502: `U_post_bump = snapshot_U(ctx_anchor["U"])` — captures
   the FINAL state, which IS at `stern_final_v`.
4. Line 503-514 builds the anchor with `U_snapshot=U_post_bump`.

So the slide15 demo correctly snapshots POST-bump U. Note: the
demo does NOT use `extract_preconverged_anchor` — it builds the
`PreconvergedAnchor` directly. That bypasses the
`result.converged` check, but the demo checks
`anchor_converged` separately at line 412 and bails on failure.

**If a caller WERE to use `extract_preconverged_anchor(result, ...)`
after Stage-2 bumping**: the original `result.U_data` is FROZEN at
Stage-1 success (captured by `solve_anchor_with_continuation`'s
return). It does NOT reflect Stage 2's post-bump state. The caller
would silently get the pre-bump U.

SEVERITY: MEDIUM — `extract_preconverged_anchor` semantics are
"extract from result", not "extract from live ctx". A future
caller who runs Stage-1 + Stage-2 and then calls extract_... with
the OLD result will silently get pre-bump U. The slide15 demo
sidesteps this by direct-building the anchor.

TRIGGER: any new caller that uses `extract_preconverged_anchor`
after a Stage-2 Stern bump (or any post-Stage-1 ctx mutation).

EVIDENCE: `anchor_continuation.py:231` —
`U_snapshot=tuple(np.asarray(arr).copy() for arr in result.U_data)`
where `result.U_data` was captured at line 1611 BEFORE Stage 2.

VERDICT: Latent footgun but not active in slide15 demo.

---

## E. AdaptiveLadder first-rung failure

LOCATION: `anchor_continuation.py:864-883`

**`_idx == 0` (first-rung failure): HANDLED CORRECTLY.**

Two paths:

1. `warm_start_floor is None` (default, used by k0 ladder): returns
   `False` → caller raises `LadderExhausted`. Geometric midpoint
   undefined when no `prev` exists.
2. `warm_start_floor is float` (opt-in, used by λ ramps at v=warm_start):
   arithmetic midpoint `0.5 * (floor + scale)`. Guard at line 879
   ensures `floor < mid < scale` (handles numerical-floor collapse).

The k0 ladder in `_run_k0_ladder` does NOT pass `warm_start_floor`,
so first-rung k0 failure correctly raises `LadderExhausted`.

The optional `warm_start_floor` was added for λ-from-warm-start
(Phase 6β step 9.5) where the "previous" state is the warm-start
(λ=0), so arithmetic bisection is the right interpolation.

VERDICT: CORRECT.

---

## F. visit_order at distance 0 (anchor V already in grid)

LOCATION: `grid_per_voltage.py:1039-1110`

**Anchor V ∈ grid: warm_walk_phi runs with v_anchor == v_target.**

Trace: visit_order sorted by `|phi - anchor.phi|`. First point's
distance is 0 if anchor V is in the grid (typical for the slide15
demo where the anchor is one of the grid points).

For that point:
- `src_phi, src_snap = anchor` (only entry in `sources` at first
  iter).
- `_restore_U` puts U at anchor state.
- `paf.assign(src_phi)` == anchor V.
- `warm_walk_phi(ctx, ..., v_anchor=src_phi, v_target=target_phi)`
  with src == target.

Inside `warm_walk_phi._march`:
- `substeps = linspace(v0, v1, 9)[1:]` with v0 == v1 → all 8
  entries are == v0 == v1.
- Each substep: `paf.assign(v_sub)` (no-op since paf already at v0),
  `run_ss(150)` — Newton at the same state should converge in ~1
  iteration to plateau (or take ss_consec=4 iterations to declare
  steady).

So 8 trivial SS solves. **Wasted work but CORRECT.**

After `_march` returns True: line 308-310 `paf.assign(v_target)` (no
change) + `run_ss(200)` (200 max steps but should plateau quickly).

Cost: ~10-20 cheap Newton steps. Acceptable.

SEVERITY: PASS (efficiency suboptimal at 0-distance, but not a bug).

VERDICT: CORRECT.

---

## G. Stern bump via _last_solver / Constant live-update

LOCATION: `forms_logc_muh.py:667-668`, `anchor_continuation.py:444-468`

**Constant live-update via _last_solver.solve(): WORKS.**

Confirmed structure:

1. `forms_logc_muh.py:667`: `stern_coeff = fd.Constant(float(
   stern_capacitance_model))`. A real `fd.Constant`, NOT a Python
   float.
2. Line 668: `F_res -= stern_coeff * (phi_applied_func - phi) * w *
   ds(electrode_marker)`. `stern_coeff` is baked into the
   `F_res` UFL form by reference (UFL captures Coefficient objects,
   not their numeric values).
3. Line 863 (in dict): `"stern_coeff_const": stern_coeff` — ctx
   exposes the live Constant.
4. `set_stern_capacitance_model` at `anchor_continuation.py:467`
   calls `stern_const.assign(nondim_value)` — mutates the Constant
   in-place. UFL form re-evaluation now uses the new value.
5. `ctx["_last_solver"].solve()` triggers PETSc SNES which
   re-assembles F_res with the updated Constant.

**Firedrake's standard pattern: `fd.Constant` is mutable; reassembly
picks up the new value.** This is the documented mechanism.

SEVERITY: PASS.

VERDICT: CORRECT — design pattern matches Firedrake semantics.

---

## H. observables.py n_e/N_ELECTRONS_REF weighting

LOCATION: `observables.py:106-121`, `_bv_common.py:213-226`

**Electron-weighted accounting: CORRECT given external scale.**

Form returns: `scale_const * Σ_j (n_e_j / 2) * R_j * ds`.

External `scale = ±I_SCALE = ±(2 * F * D_ref * c_scale / L_ref *
0.1)`. So assembled value = `±(2*F * D/L * c_scale * 0.1) *
Σ_j (n_e_j/2) * R_j * |∂Ω_electrode|` = `±F * D/L * c_scale * 0.1 *
Σ_j n_e_j * R_j * ds`. Per Faraday's law, current density = `Σ_j
n_e_j * F * R_phys_j`. The `D/L * c_scale * 0.1` factors are the
nondim → mA/cm² conversion. **The (n_e/2) form factor correctly
unfolds when external I_SCALE provides the 2*F prefactor.**

For uniform 2e reactions: `(n_e/2) = 1` for all j → reduces to
unweighted `Σ R_j`. Matches comment at line 11-15. **CONSISTENT.**

VERDICT: CORRECT.

---

## I. gross_h2o2_current uses R_2e (NOT R_4e)

LOCATION: `observables.py:123-135`

**reaction_index=0 default + slide15 demo passes 0: BOTH MAP TO R_2e.**

Confirmed reaction order in `_bv_common.py:842-865`:
- index 0 = R_2e (E°=0.695 V, 2e → H₂O₂)
- index 1 = R_4e (E°=1.23 V, 4e → H₂O)

Observable code at line 129: `idx = 0 if reaction_index is None
else int(reaction_index)`. Default 0 ⇒ R_2e ⇒ gross H₂O₂ production
rate. **CORRECT.**

Slide15 demo at `solver_demo_slide15...py:532`:
```
_build_bv_observable_form(ctx, mode="gross_h2o2_current",
                          reaction_index=0, ...)
```
Explicit `0` → R_2e. **CORRECT.**

VERDICT: CORRECT.

---

## NEW Q1: Firedrake form caching / kernel sharing

**Two ctx with structurally-identical forms but different Constant
values DO share JIT-compiled kernels** (Firedrake/TSFC caches by
form signature hash, which excludes Constant numeric values —
Constants are passed as scalar coefficients at assemble-time).

This is the foundation that makes the Stern bump pattern fast:
mutating `stern_coeff` does NOT trigger a re-compile. PyOP2 just
re-reads the Constant value.

**No correctness implication.** Cache key is the UFL form
**structure**, not the Constant values. Two contexts that build
identical-structure forms will share kernels — performance benefit
only.

VERDICT: PASS (informational).

---

## NEW Q2: Module-level mutable state leaks

Searched `Forward/bv_solver/*.py` for `global`, module-level
mutable singletons, registries. Findings:

- `forms_logc.py:316`, `forms_logc_muh.py:381`:
  `E_eq_model_global = fd.Constant(...)` — but this is a
  **LOCAL** variable inside `build_forms_logc[_muh]`, not module
  level. Name is misleading; it's per-build.
- No module-level Constant registries, no mutable singletons.
- Logger handlers and PyOP2/TSFC cache directories are configured
  via env vars (per CLAUDE.md: `MPLCONFIGDIR=/tmp` etc.) — those
  are filesystem caches, deterministic and process-local.
- `cation_hydrolysis.py` Γ state lives in `ctx["cation_hydrolysis"]`
  bundle, scoped to the context.

**No module-level mutable state that could leak between factor
iterations** (e.g., across the 4 K0_R4e factors in the slide15
demo). Each iteration rebuilds ctx + forms fresh.

VERDICT: PASS.

---

## NEW Q3: NaN propagation through summary statistics

LOCATION: `solver_demo_slide15...py:560-564, 517-518`

**NaN handled correctly at JSON serialization, but in-memory arrays
contain NaN.**

`cd_arr = np.full(NV, np.nan)` (line 517) — initialized to NaN.
`_grab` writes only on convergence. Non-converged points stay NaN.

`_to_json_list` at line 560-564:
```python
[float(x) if (np.isfinite(x) and converged[i]) else None
 for i, x in enumerate(arr)]
```
Double-guard: both `isfinite(x)` AND `converged[i]`. NaN → None.
Inf → None. **CORRECT.**

JSON output uses `None` (parses to `null`). Downstream consumers
of the JSON must handle nulls.

**However**, `cd_arr` and `pc_arr` themselves are NOT consumed for
mean/max/min in the slide15 demo — they're only used for the
JSON serialization. So NaN propagation is contained.

Plotting/summary scripts that import the JSON and compute
statistics MUST use `np.nanmean`/`np.nanmax` or explicit `None`
filtering. The slide15 plot script is OUT OF SCOPE for this audit
but worth flagging if it computes means/extrema on these arrays.

VERDICT: PASS at the demo level. CAVEAT for downstream.

---

# VERDICT

Forward solver Stage 1/2/3 algorithmic correctness on the
audited paths: **PASS with 1 LATENT FOOTGUN.**

Summary:

- (A) AdaptiveLadder: terminates in practice, but the geometric
  branch lacks an explicit `prev < mid < scale` guard. **LOW**
- (B) warm_walk_phi recursion: `v_prev_substep` tracking is
  intentionally separated from `paf` to prevent paf rollback from
  corrupting the bisection midpoint. **CORRECT**.
- (C) paf default: build-time = target V, then overwritten to src V
  before walk. **CORRECT**.
- (D) extract_preconverged_anchor: snapshots `result.U_data`
  captured at Stage 1 success — DOES NOT auto-reflect Stage-2
  bump mutations. The slide15 demo correctly direct-builds the
  anchor post-bump, sidestepping this. **MEDIUM LATENT FOOTGUN**
  for future callers that try to use `extract_preconverged_anchor`
  after Stage-2 mutations.
- (E) First-rung failure with no warm_start_floor: returns False →
  `LadderExhausted`. **CORRECT**.
- (F) Anchor V already in grid: 8 trivial SS substeps execute (no-op
  but cheap). **CORRECT**.
- (G) Stern bump via live Constant + `_last_solver.solve()`:
  Firedrake-standard pattern. **CORRECT**.
- (H) n_e/N_ELECTRONS_REF weighting in current_density: correctly
  unfolds the external I_SCALE prefactor. **CORRECT**.
- (I) gross_h2o2_current → R_2e: correct reaction in both
  preset orderings (R_2e is index 0). **CORRECT**.
- NEW1: Firedrake form caching is Constant-value-invariant.
  Performance-only, no correctness implication. **PASS**.
- NEW2: No problematic module-level mutable state.
  **PASS**.
- NEW3: NaN propagation: contained at the JSON serialization layer
  via double-guard `isfinite() and converged[i]`. **PASS** (caveat
  for downstream consumers).

**One additional minor observation from (B)**: each level of
warm-walk bisection uses `n_substeps=8` fresh substeps over a
1/8-sized interval. At max `bisect_depth=5`, this is 8^6 = 262144
potential substeps in the worst case. The depth cap is what
controls runaway cost; per-substep SS termination is bounded by
`max_ss_steps_per_substep=150` Newton solves, so the actual cap is
~40M Newton solves per warm walk. In practice, the recursion only
deepens on genuinely hard transitions and most bisections succeed
at depth ≤ 2. The cost is acceptable for the slide15 demo (V_T
overpotential changes are smooth on this stack).

No high-severity correctness defects discovered.
