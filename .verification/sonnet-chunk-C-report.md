# Chunk C Verification Report — Continuation + Grid Walk + Observables

Verified by: claude-sonnet-4-6  
Date: 2026-05-13  
Scope: `Forward/bv_solver/anchor_continuation.py`, `grid_per_voltage.py`, `observables.py`  
Doc under test: `docs/solver/forward_codepath_demo_slide15.md`

---

## Claim-by-Claim Findings

---

### Claim 1 — `anchor_continuation.py:902` is `solve_anchor_with_continuation`

**VERDICT: PASS**

Line 902 is the def of `solve_anchor_with_continuation`. Signature matches the doc
exactly: `sp_anchor, mesh, k0_targets, initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
max_inserts_per_step=4, ic_at_target=True`. Additional optional parameters
(`kw_eff_ladder`, `c_s_ladder`, `lambda_hydrolysis_ladder`, etc.) exist but do not
contradict the documented subset.

---

### Claim 2 — `anchor_continuation.py:719` is `AdaptiveLadder`; sqrt-midpoint insertion

**VERDICT: PASS with annotation**

Line 719 is `class AdaptiveLadder`. Methods `is_done()`, `current_scale`,
`record_success()`, `record_failure_and_insert()` all exist and behave as described.

The insertion rule (line 884):
```python
midpoint = math.sqrt(prev * scale)
```
This is the **geometric mean** (sqrt of product), not arithmetic midpoint. The doc
says "sqrt-midpoint" which is accurate for post-first-rung failures.

**Annotation (not an error):** For *first-rung* failures when `warm_start_floor` is
set, the ladder uses the **arithmetic midpoint** (`0.5 * (warm_start_floor + scale)`)
because `sqrt(0 * x) = 0` is degenerate. The doc does not mention this dual-mode
behavior, but the claim "Sqrt-midpoint insertion recovers from failed rungs" is
accurate for the common (non-first-rung) case and is the Layer 1 mechanism described.

---

### Claim 3 — `anchor_continuation.py:246` is `solve_reaction_k0_model`

**SEVERITY: warning**  
**LOCATION: anchor_continuation.py:246**

The doc states line 246 is `solve_reaction_k0_model`. The actual function at line 246
is `set_reaction_k0_model` (not `solve_reaction_k0_model`). The function `solve_reaction_k0_model`
does not exist anywhere in `anchor_continuation.py`.

The Stage 1 loop (line 1195) calls `set_reaction_k0_model(ctx, j, k0_scale * k_target)`,
and the file summary table at the end of the doc also lists `solve_reaction_k0_model`
under Stage 1. Both references use the wrong function name.

**Evidence:**
```
anchor_continuation.py:246  def set_reaction_k0_model(ctx: dict, j: int, k0_model_value: float) -> None:
anchor_continuation.py:1195     set_reaction_k0_model(ctx, j, k0_scale * k_target)
```

The behavior described is correct; only the name is wrong.

---

### Claim 4 — `set_stern_capacitance_model` mutates `ctx['stern_capacitance_func']`

**SEVERITY: warning**  
**LOCATION: anchor_continuation.py:448**

The doc claims:
```
• mutates ctx['stern_capacitance_func']  ← shared FE Constant
```

The actual code reads and assigns to `ctx['stern_coeff_const']`:
```python
stern_const = ctx.get("stern_coeff_const")   # line 448
...
stern_const.assign(nondim_value)              # line 467
```

The ctx key is `'stern_coeff_const'`, not `'stern_capacitance_func'`. The doc has
the wrong key name. The functional description (shared FE Constant, residual sees
new C_S without rebuilding forms) is correct.

---

### Claim 5 — Stage 1 ladder behavior: `for j, k_target in k0_targets`

**VERDICT: PASS with annotation**

The Stage 1 loop (lines 1192–1195) iterates:
```python
k0_scale = k0_lad.current_scale
for j, k_target in k0_targets.items():
    set_reaction_k0_model(ctx, j, k0_scale * k_target)
```

`k0_targets` is typed as `Dict[int, float]` and iterated via `.items()`, not as a
list of pairs. The doc pseudocode writes `for j, k_target in k0_targets:` (without
`.items()`), which would be incorrect Python for a dict. The intent is accurate but
the pseudocode syntax would silently fail for a dict — it would iterate keys only.

**Annotation:** The doc claims `k0_targets` is iterable as `(j, k_target)` pairs
directly. This is only true for a list of tuples; the actual type is a dict requiring
`.items()`. The pseudocode is misleading but the described behavior is correct.

---

### Claim 6 — Stage 2 Stern bump: `ctx['_last_solver'].solve()` reuses anchor solver

**SEVERITY: critical**  
**LOCATION: docs/solver/forward_codepath_demo_slide15.md:106, anchor_continuation.py:1271–1317**

The doc describes Stage 2 as:
```
for cs in _stern_bump_ladder(target):
    set_stern_capacitance_model(ctx, cs)
    ctx['_last_solver'].solve()   ← reuse the anchor's solver
```

The actual code (lines 1271–1317) runs a **full `_run_k0_ladder`** at each C_S rung
— not a bare `ctx['_last_solver'].solve()`:
```python
for cs_val in cs_seq:
    set_stern_capacitance_model(ctx, cs_val)
    ok, k0_lad, _ = _run_k0_ladder(f"cs={cs_val:.3e}")  # full k0 ladder
```

This means each C_S rung re-runs the entire k0 continuation ladder (1e-12 → 1.0)
from the last successful state, which is far more expensive and robust than a single
Newton solve. The doc fundamentally misrepresents Stage 2's convergence mechanism and
cost.

**Additional issue in Claim 6:** The doc also says the output of Stage 2 is:
```
PreconvergedAnchor(phi_eta, U_snapshot, k0_targets, dof_count, ladder_history)
```
But the actual field name is `phi_applied_eta` not `phi_eta`:
```python
class PreconvergedAnchor:
    phi_applied_eta: float     # line 133
```
The constructor call in `extract_preconverged_anchor` uses `phi_applied_eta=...`.
The doc abbreviates this to `phi_eta` which is not a valid field name.

---

### Claim 7 — `grid_per_voltage.py:875` is `solve_grid_with_anchor`; sorted by |V − V_anchor|

**VERDICT: PASS**

Line 875 is `solve_grid_with_anchor`. The sorting (lines 1039–1044):
```python
visit_order = sorted(
    range(n_points),
    key=lambda i: abs(float(phi_applied_values[i]) - float(anchor.phi_applied_eta)),
)
```
This sorts by `|phi - anchor.phi_applied_eta|` ascending — closest first. Matches doc.

For each target, the nearest converged source is chosen (lines 1077–1081), U is
restored, `ctx['phi_applied_func'].assign(src_phi)` is called (line 1087), then
`warm_walk_phi` is called. All described steps verified correct.

---

### Claim 8 — `grid_per_voltage.py:218` is `warm_walk_phi`; parameter `bisect_depth_warm`

**SEVERITY: warning**  
**LOCATION: grid_per_voltage.py:218–235**

Line 218 is `warm_walk_phi`. However the doc claims the signature includes
`bisect_depth_warm` as a parameter name:
```
warm_walk_phi(ctx, solver, of_cd, v_anchor_eta, v_target_eta, n_substeps, bisect_depth_warm)
```

The actual parameter is named `bisect_depth` (not `bisect_depth_warm`):
```python
def warm_walk_phi(
    *,
    ctx,
    solver,
    of_cd,
    v_anchor_eta: float,
    v_target_eta: float,
    n_substeps: int = 4,
    bisect_depth: int = 3,        # ← not bisect_depth_warm
    ...
```

`bisect_depth_warm` is the name of the *outer* parameter in `solve_grid_with_anchor`
(line 882) that gets passed as `bisect_depth=bisect_depth_warm` to `warm_walk_phi`
(line 1096). The doc conflates the two parameter names.

The `_march` logic (arithmetic midpoint bisection, `depth >= bisect_depth` check,
recursion) and the final pin (`paf.assign(tgt); return run_ss(max_steps_final)`) all
match the doc description exactly.

The 32× refinement claim (`bisect_depth=5` → 2^5 = 32 minimum interval divisions) is
mathematically correct. The default `bisect_depth` in `warm_walk_phi` is 3 (not 5),
but `solve_grid_with_anchor` passes `bisect_depth_warm=5`, so the Layer 8 description
("up to bisect_depth_warm=5 levels (32× refinement)") correctly refers to the outer
function's default.

---

### Claim 9 — `grid_per_voltage.py:135` is `make_run_ss`; SER plateau detection on `j_cd`

**VERDICT: PASS**

Line 135 is `make_run_ss`. The closure (line 193):
```python
fv = float(fd.assemble(of_cd))
```
The observable assembled is `of_cd` — the current-density form built by
`_build_bv_observable_form(ctx, mode="current_density", ...)`. The variable is
named `fv` internally (not `j_cd`), but semantically it is the cathodic current
density. The doc calling it `j_cd` is a naming convention, not an error.

The plateau logic:
```python
is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)
steady_count = steady_count + 1 if is_steady else 0
if steady_count >= ss_consec:
    return True
```
The condition `|Δj_cd| < ss_rel_tol · |j_cd| + ss_abs_tol` in the doc corresponds to:
- `delta / sv <= ss_rel_tol` where `sv = max(|fv|, |prev_flux|, ss_abs_tol)` (relative)
- `delta <= ss_abs_tol` (absolute floor)

This is slightly more conservative than the doc's simplified expression (`sv` uses the
max of current and previous flux, not just current `|j_cd|`), but the description is
accurate as an approximation. Not an error.

The SER dt adaptation (grow if `prev_delta / delta > 1` else shrink) matches lines
199–206.

---

### Claim 10 — `observables.py:67` is `_build_bv_observable_form`; `n_e/2` weighting

**VERDICT: PASS with annotation**

Line 67 is `_build_bv_observable_form`. Modes `"current_density"` and
`"gross_h2o2_current"` exist and behave as described.

For `"current_density"` (lines 110–121):
```python
weight = fd.Constant(float(n_e_j) / ref)    # where ref = N_ELECTRONS_REF = 2
rate_sum = rate_sum + weight * R_j
```
The weight is `n_e_j / 2`. For R_2e (n_e=2) this gives weight=1.0; for R_4e
(n_e=4) this gives weight=2.0. The doc notation `Σ_j (n_e/2)·R_j` is correct.

**Annotation on the doc's description:** The doc says `(n_e/2)·R_j` — this is
physics-accurate notation. `N_ELECTRONS_REF = 2` is anchored to the `I_SCALE`
calibration in `_bv_common.compute_i_scale` (which assumes n_e=2 as the reference).
This is not a division by 2 for an RRDE collection factor — it is the electron-count
normalization so that the 4e reaction contributes 2× the current of the 2e reaction,
consistent with Faraday's law. The notation in the doc is correct but could be
misread as RRDE-specific; it is not.

For `"gross_h2o2_current"` (line 135):
```python
return scale_const * bv_rate_exprs[idx] * ds(electrode_marker)
```
This is `∫ R_R2e ds` (single reaction, no weighting). Doc says `∫ R_R2e ds only` —
correct.

---

### Convergence-Mechanism Table

| Layer | Doc claim | Verdict |
|---|---|---|
| 1 | k0 ladder, AdaptiveLadder @ :719, sqrt-midpoint | PASS |
| 5 | Stern bump via set_stern_capacitance_model | WARNING: mutates `stern_coeff_const` not `stern_capacitance_func`; Stage 2 runs full k0 ladder per rung, not bare `.solve()` |
| 6 | SER plateau, make_run_ss @ :135, surface flux | PASS |
| 7 | Warm-start from nearest neighbor @ :875 | PASS |
| 8 | φ-substep bisection @ :218, bisect_depth_warm=5 | WARNING: param is `bisect_depth` in warm_walk_phi, not `bisect_depth_warm` |

---

## Summary of Issues

| # | Severity | Location | Issue |
|---|---|---|---|
| A | warning | doc line 73, doc file-list | Function named `solve_reaction_k0_model` does not exist; correct name is `set_reaction_k0_model` (anchor_continuation.py:246) |
| B | warning | doc line 104 | `ctx['stern_capacitance_func']` is wrong key; actual key is `ctx['stern_coeff_const']` |
| C | critical | doc lines 102–106 | Stage 2 description is fundamentally wrong: each C_S rung calls `_run_k0_ladder()` (full k0 ladder), NOT `ctx['_last_solver'].solve()`. Wall time and convergence behavior differ significantly. |
| D | warning | doc line 108 | `PreconvergedAnchor` field shown as `phi_eta`; actual field is `phi_applied_eta` |
| E | warning | doc line 139 | `warm_walk_phi` signature shows `bisect_depth_warm` as a parameter; actual parameter name is `bisect_depth` |
| F | note | doc line 72 | Pseudocode `for j, k_target in k0_targets:` would fail on a dict; should be `k0_targets.items()` |

---

## VERDICT: ISSUES FOUND

One critical issue (Stage 2 mechanism completely misstated) and four warnings. All
line number annotations for function definitions are correct. The convergence
descriptions for Layers 1, 6, 7, and 8 are accurate. The Stage 2 description and
several minor naming/key discrepancies require correction.
