# Verification Report — Run 2 (post-fix)

**Target:** Same codepath as run 1 (`scripts/plot_iv_curve_unified.py`), after applying the three fixes from `.verification/REPORT.md` to `Forward/bv_solver/grid_per_voltage.py`.
**Date:** 2026-05-02
**Level:** 1 (Sonnet only)
**Scope:** 15 files, ~4360 lines, 3 subsystems.
**Agents:** 5 Sonnet agents (same chunk split as run 1).
**Verdict:** **PASS** — all three previously-flagged bugs confirmed CLOSED. One new minor surfaced inside my own fix code; corrected post-report.

---

## Summary

The C+D orchestrator patches close the two critical bugs (degenerate `_march` bisection, silent SNES non-convergence) and the major (`solve_opts` sub-dict pollution) found in run 1. Independent re-verification by all 5 chunks finds no critical or major issues anywhere on the codepath. Run 2 surfaced 8 additional minor/cosmetic items across the codebase — none of them affect production correctness, and most are cleanups that future maintainers can pick up incrementally.

---

## Closed bugs (confirmed by chunk 4 re-verification)

| Bug | File:line | Status |
|---|---|---|
| Critical E (silent SNES non-convergence) | `grid_per_voltage.py:240` | **CLOSED** — `setdefault("snes_error_if_not_converged", True)` causes Firedrake to raise `ConvergenceError`; `run_ss`'s `except Exception: return False` catches it. |
| Critical H (degenerate `_march` bisection) | `grid_per_voltage.py:352-384` | **CLOSED** — `v_prev_substep` tracked explicitly across iterations; `paf.assign(v_prev_substep)` after every `_restore_U` on failure path makes `v_mid` non-degenerate. Verified through first-substep failure, mid-loop failure, and nested bisection traces. |
| Major D (`solve_opts` sub-dict pollution) | `grid_per_voltage.py:228-235` | **CLOSED** — `_NON_PETSC_KEYS` filter correctly excludes `bv_bc`, `bv_convergence`, `nondim`, `robin_bc` while preserving every PETSc option. |

---

## New findings (run 2)

| # | Severity | Location | Issue | Status |
|---|----------|----------|-------|--------|
| N1 | minor (my fix) | `grid_per_voltage.py:228-235` | `isinstance(params, dict)` guard sat inside the comprehension's filter clause, so `.items()` on `(params or {})` would still raise `AttributeError` on a non-dict, non-None `params`. Not reachable in production, but the defensive intent was unfulfilled. | **FIXED in this turn** — moved the `isinstance` check ahead of `.items()`: `_items = params.items() if isinstance(params, dict) else []`. |
| N2 | minor (cosmetic) | `grid_per_voltage.py` Phase 2 failure branches (~lines 492, 532) | On warm-walk failure, `points[orig_idx]` retains `method="cold-failed"` from Phase 1 instead of being updated to `"warm-failed"`. Diagnostic only. | open |
| N3 | minor | `scripts/_bv_common.py:309-321` (`_make_bv_convergence_cfg`) | Factory does not emit `packing_floor`. `_get_bv_convergence_cfg` falls back to default `1e-8`; `_default_bv_convergence_cfg` does include it. Inconsistency between factory and default. | open |
| N4 | minor | `Forward/bv_solver/forms_logc.py:318-323` | `if E_eq_j_val is not None and E_eq_j_val != 0.0` would silently fall through to `E_eq_model_global` for a reaction legitimately at E_eq = 0 V (SHE reference). Not exercised by production (R1=0.68, R2=1.78). | open (was prior E7 in run 1) |
| N5 | minor | `Forward/bv_solver/forms_logc.py:448-473` | `ctx.update` omits `bv_scaling` key that `forms.py` exposes. Downstream code keying on `ctx["bv_scaling"]` raises `KeyError` against a logc context. | open |
| N6 | minor | `Forward/bv_solver/boltzmann.py:110` | `c_bulk_val == 0.0` exact float check silently skips zero-concentration entries; the return count overcounts skipped entries. Not exercised by production (`c_bulk_nondim=0.2`). | open |
| N7 | minor | `Forward/bv_solver/config.py:214-215` | `reactions: []` (empty list) returns `None` with no warning, silently falling back to legacy single-reaction BV. Could mask YAML misconfiguration. | open |
| N8 | minor | `Nondim/transform.py:103` (`_pos`) | `float(value)` is unguarded — non-numeric input raises `TypeError` instead of `ValueError`, inconsistent with `_as_list` and `_bool`. Zero production impact. | open |
| N9 | minor | `Nondim/transform.py:459-481` (`verify_model_params`) | Uses `print()` instead of `warnings.warn()`, bypassing `warnings.filterwarnings`. | open |

---

## Re-confirmed prior findings (run 1)

All majors and minors from run 1 that were OUT OF SCOPE for the fix landed remain open:

| # | Severity | Location | Status |
|---|----------|----------|--------|
| 4 | major | `Forward/bv_solver/config.py:24-26` (`_get_bv_cfg`) | When `alpha` is a list/tuple, the `(0,1]` range check is skipped. Production uses per-reaction validation, but legacy path is unguarded. | open |
| 5 | major | `Forward/bv_solver/config.py:259-260` (`_get_bv_reactions_cfg`) | `cathodic_species` / `anodic_species` not bounds-checked against `[0, n_species)`. | open |
| 6 | minor | `scripts/plot_iv_curve_unified.py:220` | CSV writes `nan` for missing `z_factor` instead of `""`. | open |
| 7 | minor | `Forward/params.py:90-100` | `__setitem__` bypasses `@dataclass(frozen=True)` via `object.__setattr__`. | open |
| 8 | minor | `Forward/bv_solver/forms_logc.py:349-357` | Log-rate `else: anodic = Constant(0.0)` covers both irreversible and `reversible=True, c_ref ≤ 1e-30`. | open |
| 10 | minor | `Forward/bv_solver/config.py:80-93` | `conc_floor=1e-8` in defaults vs `1e-12` in production factory. | open |

---

## Items confirmed CLEAN (full details in chunk reports)

- Sign conventions on CD/PC (chunks 1, 4).
- Boltzmann residual sign + bulk neutrality (chunks 1, 3).
- z-ramp invariant: both `z_consts[i]` and `boltzmann_z_scale` zeroed (chunks 2, 4).
- Log-rate ↔ non-log-rate algebraic equivalence (chunk 2).
- Nondim prefactors numerically verified: `electromigration_prefactor=1.0` exactly, `poisson_coefficient=3.7e-8`, `charge_rhs_prefactor=1.0` (chunk 5).
- I_SCALE = 0.1833 mA/cm² (chunks 1, 5).
- E_eq nondimensionalization (chunks 2, 3).
- Mesh marker chain 3/4/4 (chunk 1).
- `_clone_params_with_phi` (chunk 4) — unchanged, correct.
- `adj.stop_annotating()` integrity (chunks 1, 4).
- `per_point_callback` contract honored at all three call sites (cold success, cathodic warm success, anodic warm success) (chunk 4).
- `_solve_cold` z-ramp orchestration (chunk 4) — unaffected by the `_march` fix because `paf` is set once at line ~305 and not mutated inside the z-ramp loop.
- `_restore_U` snapshot/restore semantics for `U` and `U_prev` (chunks 1, 4).
- `ckpt_inner` / `ckpt_outer` timing in the patched `_march` (chunk 4).
- Steric (Bikerman) well-posedness at the bulk Σ a c ≈ 0.012 (chunk 2).
- Boltzmann phi_clamp=50 reachability: production max `|phi_hat| ≈ 23.4 < 50` (chunk 3).
- `phi_clamp` Jacobian linearization correctness (chunk 3).

---

## Suggested follow-up sweeps (optional, non-blocking)

1. **Bug 3 propagation across the codebase.** The same `solve_opts = dict(params)` pattern (without sub-dict filtering) lives in 4 sites in `Forward/bv_solver/solvers.py` (`forsolve_bv:68`, `solve_bv_with_continuation:151`, `solve_bv_with_ptc:354`, `solve_bv_with_charge_continuation:531`) and 1 site in `Forward/bv_solver/grid_charge_continuation.py:183`. None of them are on this script's codepath, but the `_NON_PETSC_KEYS` filter would close the same noise everywhere. Single-batch sweep, ~5 small edits.
2. **Major 4 / Major 5** (config.py validation gaps) — small one-line guards.
3. **N2** (Phase 2 cosmetic) — update `method=` field on warm-walk failure.

---

## Pointers

- Run 1 report: `.verification/REPORT.md`
- Run 2 per-chunk reports: `.verification/run2/sonnet-chunk-{1..5}-report.md`
- Codepath map: `docs/plot_iv_curve_codepath.md`
