# Edit-Batch Verification Report

**Date:** 2026-05-05
**Auditor:** claude-sonnet-4-6
**Diff:** `/tmp/bv_changes_diff.txt` (446 lines)
**Prior report:** `.verification/REPORT.md` (issued 2026-05-05, 26 findings)
**Files examined:** `Forward/bv_solver/forms_logc.py`, `forms_logc_muh.py`, `validation.py`, `grid_per_voltage.py`, `config.py`, `FluxCurve/bv_point_solve/__init__.py`, `bv_point_solve/forward.py`, `scripts/_bv_common.py`, `Forward/bv_solver/nondim.py`

---

## 8-Row Summary

| # | Change | Status | Evidence |
|---|--------|--------|----------|
| 1 | `exponent_clip` IC fallback 50→100 in both logc files | **PASS** | `forms_logc.py:702`, `forms_logc_muh.py:800` both read `100.0` |
| 2 | Phase 2 interior gap-fill in `grid_per_voltage.py` | **PASS with note** | Lines 612–696; logic correct — see notes below |
| 3 | FluxCurve callers pass `mu_species`, `em`, `reaction_e_eq`, `bv_exp_scale` | **PASS** | `__init__.py:714–733`, `forward.py:309–328` |
| 4 | W1 clip-saturation warning implemented in `validation.py` | **PASS with note** | `validation.py:202–216`; sign convention caution noted |
| 5 | W5 DOF-mask alignment fix in `validation.py` | **PASS** | `validation.py:223–257` |
| 6 | `_validate_formulation` warns on `"concentration"`, defaults → `"logc"` | **PASS** | `config.py:82–105`, `143–147`, `162` |
| 7 | Per-element alpha validation when alpha is list/tuple | **PASS** | `config.py:28–39` |
| 8 | `cathodic_species`/`anodic_species` bounds-check; stored vars are validated | **PASS** | `config.py:316–331`, `369–370` |

**Overall verdict: PASS.** All 8 changes are correctly implemented. Two minor notes (below) do not constitute regressions.

---

## Detailed Findings

### Change 1 — `exponent_clip` fallback 50→100

PASS. `forms_logc.py:702` and `forms_logc_muh.py:800` both now read `float(conv_cfg.get("exponent_clip", 100.0))`. No other `50.0` magic literal exists in either IC helper. `u_clamp` and `phi_clamp` fallbacks in those functions were intentionally left at their prior values per user request and are not affected.

Addresses prior-report Finding #1.

### Change 2 — Phase 2 interior gap-fill

PASS. Correctness trace:

- Guard `if anchor_hi - anchor_lo > 1` (line 612): when `anchor_lo == anchor_hi` (single cold success) the difference is 0, skipped. When `anchor_hi == anchor_lo + 1` (adjacent pair) the difference is 1, also skipped. `range(anchor_lo+1, anchor_hi)` is empty in both cases — no-op as specified.
- `j_lo` scan: starts at `orig_idx - 1`, walks down while unconverged and `>= 0`; if it reaches `< 0` the post-loop guard sets `j_lo = None`. Correct.
- `j_hi` scan: starts at `orig_idx + 1`, walks up while unconverged and `< n_points`; if it reaches `>= n_points` the post-loop guard sets `j_hi = None`. Correct.
- Both `j_lo is None and j_hi is None` → `continue` (no anchor available). Correct.
- `failed_with_anchors` skip: identical `anchor_key` on a subsequent pass means no new converged neighbour became available → correctly skipped without retry.
- On success: `snapshots[orig_idx] = snap` (line 662) is set **before** `points[orig_idx]` is updated (line 673), so a subsequent pass that picks this point as anchor finds a valid snapshot. Correct.
- On failure: marks `failed_with_anchors[orig_idx] = anchor_key`, does **not** break — continues to next point. Correct (original cathodic/anodic outer walks still `break` on first outer failure; the interior loop correctly does not).
- `phase="warm_walk_interior"` in `collect_diagnostics` (line 668). Correct.
- `method = f"warm<-{phi_applied_values[j]:+.3f}"` (line 663). Correct.
- `ctx` reassignment pattern (line 656) is identical to the cathodic/anodic walks (lines 519, 567) — not a new regression.
- **Snapshot null-safety:** `j_lo`/`j_hi` are only used if `points[j].converged is True`. A converged point always has `snapshots[j] is not None` (invariant maintained by Phases 1 and 2: snapshot is set before `converged=True` in every code path). No `None` snapshot can reach `_solve_warm`.

**NOTE (non-regression):** When all interior failures share both anchors `(j_lo, j_hi)` across all passes (no new converged points ever appear), `failed_with_anchors` entries skip all re-attempts and `made_progress` stays False → outer while-loop exits cleanly. Correct. However, if an interior point converges on pass N but then would serve as anchor for a neighbour on pass N, it won't: the inner loop visits points in `range(anchor_lo+1, anchor_hi)` order and a point converged mid-pass won't be reflected in the j_lo/j_hi scans for points already skipped this pass. That's acceptable — the **next** `while made_progress` iteration picks it up. Minor latency but not incorrect.

Addresses prior-report Finding #10.

### Change 3 — FluxCurve callers: `mu_species`, `em`, `reaction_e_eq`, `bv_exp_scale`

PASS. Both call sites verified:

- `_scaling_v = ctx.get("nondim", {})` — correct; `bv_reactions`, `bv_exponent_scale`, and `electromigration_prefactor` are all stored in the `nondim` dict by `_add_bv_reactions_scaling_to_transform` (`nondim.py:163–164`) and `_add_bv_scaling_to_transform` (`nondim.py:96`).
- `mu_species=ctx.get("mu_species")` — `forms_logc_muh.py:578` sets `ctx["mu_species"] = list(mu_species)`. `forms_logc.py` does NOT set `ctx["mu_species"]`; `ctx.get("mu_species")` returns `None` for logc contexts. `validation.py:101` handles `None` via `mu_set = frozenset()` — treated as "no muh species." Correct.
- `em=float(_scaling_v.get("electromigration_prefactor", 1.0))` — key is set in all nondim dicts (`nondim.py:96` via `bv_exponent_scale` block, and directly at `forms_logc.py:217`, `forms_logc_muh.py:249`). Fallback 1.0 matches production nondim mode. Correct.
- `_rxn_e_eq_v`: uses `r.get("E_eq_model", 0.0)`. Key `"E_eq_model"` is set in every `bv_reactions` entry by `_add_bv_reactions_scaling_to_transform` (lines 134, 144). The `or []` guard ensures an empty list produces `None` rather than an empty list (matching `reaction_e_eq is None` → W1 skipped). Correct.
- `bv_exp_scale=float(_scaling_v.get("bv_exponent_scale", 1.0))` — key is set in all nondim dicts. Fallback 1.0 is safe. Correct.
- `exponent_clip` fallback bumped 50→100 at both sites (diff lines 10, 40). Consistent.

Addresses prior-report Finding #3.

### Change 4 — W1 clip-saturation warning

PASS. Implementation at `validation.py:202–216` correctly:
- Guards on `reaction_e_eq is not None and exponent_clip > 0`.
- Uses `phi_data` already computed at line 178 (reused as `phi_min_dof`/`phi_max_dof`). No redundant read.
- `threshold = exponent_clip - w1_margin`. Default `w1_margin=1.0`.
- Iterates over reactions with index `j` for the warning message.
- Appends to `warns` list (not `fails`), consistent with W-class checks.

**NOTE — sign convention vs. non-Stern path:**
`_build_eta_clipped` in both `forms_logc.py:234–253` and `forms_logc_muh.py:293–308` uses:
- Stern enabled: `eta_raw = phi_applied - phi - E_eq`
- Stern disabled, `use_eta_in_bv=True`: `eta_raw = phi_applied - E_eq` (phi-independent)
- Stern disabled, `use_eta_in_bv=False`: `eta_raw = phi - E_eq`

W1 in the validator always computes `phi_applied - phi_{min,max} - E_eq_j`. This is exact only for the Stern path (production stack). For `use_eta_in_bv=True` (non-Stern), eta is constant across the domain at `bv_exp_scale * (phi_applied - E_eq_j)`; the validator artificially widens the range by `±Δphi`, making W1 **over-sensitive** (false positives possible) but never missing real saturation. For `use_eta_in_bv=False`, eta varies with phi but in the opposite sign from the validator's formula — W1 could misestimate the max |eta|. However, the `use_eta_in_bv=False` path is non-default (`_default_bv_convergence_cfg` sets `use_eta_in_bv=True`), and W1 is documented as a conservative early-warning. This is acceptable but should be noted.

This is a **pre-existing design limitation**, not a regression introduced by this edit.

Addresses prior-report Finding #4.

### Change 5 — W5 DOF-mask alignment fix

PASS. New code:
- Checks `any(z_vals[i] > 0 and c_bulk[i] > 0 ...)` before importing Firedrake (lazy import; correct).
- Per-species: `V_full.sub(i).collapse()` gives the correct scalar subspace for species i regardless of element order.
- `fd.SpatialCoordinate(mesh)[ydim]` interpolated onto `V_i` → y-coordinate DOF array aligned with `_conc_array(i)` for any element order. Correct.
- Fallback to `mesh.coordinates.dat.data_ro` with shape-mismatch guard (`warns.append("W5: skipped ...")`) — safe degradation.
- `ydim = mesh.geometric_dimension() - 1`: for 1D interval meshes, `geometric_dimension()=1`, so `ydim=0` (the only spatial coordinate). For 2D rectangle meshes, `ydim=1` (y-axis). Both match the production mesh conventions (electrode at bottom, bulk at top). Correct.

Addresses prior-report Finding #5.

### Change 6 — `_validate_formulation` warns on `"concentration"`, defaults → `"logc"`

PASS. Verified:
- `value is None` → returns `"logc"` (`config.py:91`). Was `"concentration"`. Fixed.
- `s == "concentration"` → emits `DeprecationWarning` with `stacklevel=2` (`config.py:97–104`). The string `"concentration"` is returned after the warning, maintaining back-compat for dispatch (which silently falls through to logc). Correct.
- `_default_bv_convergence_cfg` returns `"formulation": "logc"` (`config.py:146`). Was `"concentration"`. Fixed.
- `_get_bv_convergence_cfg` calls `_validate_formulation(raw.get("formulation", "logc"))` (`config.py:162`). Was `"concentration"`. Fixed.
- `"concentration"` remains in `_VALID_FORMULATIONS` (`config.py:78`) — old saved configs parse without ValueError; DeprecationWarning fires instead. Correct back-compat.
- `import warnings` present at top of file (`config.py:5`). Correct.

Addresses prior-report Finding #6.

### Change 7 — Per-element alpha validation

PASS. New block at `config.py:28–39`:
- Only executes when `alpha_val is None` (i.e., alpha is a list/tuple — line 28).
- Iterates with `enumerate(alpha)`, converting each entry to float; raises `ValueError` with offending index and value on range violation or non-float. Correct.
- The existing scalar path (`alpha_val is not None`) is unchanged. No regression.

Addresses prior-report Finding #7.

### Change 8 — `cathodic_species`/`anodic_species` bounds-check

PASS. New validation at `config.py:316–331`:
- `cat_idx = int(cat)` followed by `if cat_idx < 0 or cat_idx >= n_species` raises with descriptive message. Correct.
- `anod_idx` path: `None` passes through unchanged (`anod_idx = None`); otherwise same `[0, n_species)` check. Correct.
- `reactions.append({... "cathodic_species": cat_idx, "anodic_species": anod_idx, ...})` at `config.py:369–370` — uses the validated `cat_idx`/`anod_idx` variables, not the raw `cat`/`anod` strings. Correct.

Addresses prior-report Finding #8.

---

## Prior-Report Findings Status After This Edit Batch

| Prior Finding | Status |
|---|---|
| #1 — IC exponent_clip 50.0 fallback | **Fixed** (Change 1) |
| #2 — IC u_clamp 30.0 fallback | **Intentionally not changed** (user request; u_clamp and phi_clamp left at old values) |
| #3 — FluxCurve callers missing mu_species/em | **Fixed** (Change 3) |
| #4 — W1 unimplemented | **Fixed** (Change 4) |
| #5 — W5 DOF-mask alignment | **Fixed** (Change 5) |
| #6 — concentration default/whitelist | **Fixed** (Change 6) |
| #7 — alpha list range check | **Fixed** (Change 7) |
| #8 — species index bounds | **Fixed** (Change 8) |
| #9 — Boltzmann z=0 silently accepted | **Not addressed** (not in this diff) |
| #10 — Interior cold-failed gap | **Fixed** (Change 2) |
| #11 — factory phi_clamp=50.0 | **Not addressed** (not in this diff) |
| #12 — factory E_eq defaults 0.0 | **Fixed** (earlier separate commit: `_bv_common.py:318–319`, `449–450`) |
| #13–26 — notes | **Not addressed** (no rush, per prior report) |

---

## Cross-Cutting Issues and New Bugs

**None found.**

Specific cross-cutting checks performed:
1. `mu_species` name: `forms_logc_muh.py:578` sets `ctx["mu_species"]`; both call sites read `ctx.get("mu_species")`. Names match.
2. `bv_reactions` key: `nondim.py:163` stores `out["bv_reactions"]`; call sites read `_scaling_v.get("bv_reactions", [])`. Names match.
3. `bv_exponent_scale` key: `nondim.py:164` stores `out["bv_exponent_scale"]`; call sites read `_scaling_v.get("bv_exponent_scale", 1.0)`. Names match.
4. `E_eq_model` key: `nondim.py:134,144` stores `srxn["E_eq_model"]`; call sites read `r.get("E_eq_model", 0.0)`. Names match.
5. `electromigration_prefactor` key: set in `nondim.py` scaling output; read with `.get(..., 1.0)` fallback. Names match.
6. `exponent_clip` is now read at three layers: `_get_bv_convergence_cfg` (default 100.0), IC fallbacks (100.0), FluxCurve caller fallbacks (100.0). All consistent.
7. Interior fill `snapshots[orig_idx]` update precedes `points[orig_idx]` update; null-snapshot invariant maintained.
8. `"concentration"` DeprecationWarning is emitted and the value is returned unchanged, so the dispatcher still receives the string and can route it to logc. No silent swallow.

**Remaining open issues from prior report:** Findings #9, #11, and notes #13–26 are unaddressed but were assessed as low/no urgency. No new issues introduced.
