# Verification Report — `plot_iv_curve_unified.py` Codepath

**Target:** Every file live on the `scripts/plot_iv_curve_unified.py` codepath (mapped in `docs/plot_iv_curve_codepath.md`).
**Date:** 2026-05-02
**Level:** 1 (Sonnet only)
**Scope:** 15 files, ~4337 lines, 3 subsystems (`scripts/`, `Forward/`, `Nondim/`).
**Agents:** 5 Sonnet agents (chunked by cohesion).
**Chunking:**
1. `plot_iv_curve_unified.py`, `_bv_common.py`, `Forward/params.py`
2. `bv_solver/__init__.py`, `dispatch.py`, `forms_logc.py`
3. `bv_solver/config.py`, `nondim.py`, `boltzmann.py`
4. `bv_solver/grid_per_voltage.py`, `observables.py`, `mesh.py`, `solvers.py` (just `_clone_params_with_phi`)
5. `Nondim/transform.py`, `Nondim/constants.py`

**Verdict:** **ISSUES FOUND** — 2 critical bugs, 1 major, plus 2 major validation gaps and several minor/question items. Both critical bugs are independently confirmed by direct source inspection.

---

## Summary

The production 3-species + analytic Boltzmann counterion + log-c + log-rate Butler–Volmer stack is wired correctly end-to-end at the **physics** level: sign conventions, dimensionless prefactors (electromigration_prefactor=1.0, poisson_coefficient=(λ_D/L)²≈3.7e-8, charge_rhs_prefactor=1.0), Boltzmann residual sign, log-rate ↔ non-log-rate algebraic equivalence, BC routing, mesh-marker chain, I_SCALE units (0.1833 mA/cm²), z-ramp invariant, and adjoint-tape suppression all check out.

The two **critical** issues live in the C+D **orchestrator** (`grid_per_voltage.py`):

1. **Bisection in `_march` is degenerate** — `_restore_U` restores `U` and `U_prev` but not `phi_applied_func`, so on substep failure the recursion midpoint collapses to the failed voltage and the bisection cannot make progress. Any warm-walk substep that misses on first try is unrecoverable.
2. **SNES non-convergence is silently accepted** — the orchestrator deliberately omits `snes_error_if_not_converged=True` and relies on `try/except Exception` in `run_ss`. Firedrake does not raise on SNES divergence by default, so a non-converged Newton iterate can be accepted, time-stepped, and ultimately declared "steady" by the plateau detector if the divergent state happens to be flat.

Both fixes are one-liners.

---

## Findings

| # | Severity | Location | Issue | Found By |
|---|----------|----------|-------|----------|
| 1 | **critical** | `Forward/bv_solver/grid_per_voltage.py:355-356` (`_march` inside `_solve_warm`) | After substep failure, `_restore_U` does NOT restore `paf = ctx["phi_applied_func"]` (a separate R-space `Function`, not inside `U.dat`). Line 355 reads `v_prev = float(paf.dat.data_ro[0])`, which still equals `v_sub` (the just-failed value), making `v_mid = 0.5·(v_sub + v_sub) = v_sub`. The recursive `_march(v_sub, v_sub, depth+1)` calls produce zero-width intervals and immediately re-fail at the same voltage, exhausting `bisect_depth_warm` without progress. **Bisection is non-functional.** | Sonnet ch.4 |
| 2 | **critical** | `Forward/bv_solver/grid_per_voltage.py:228-234` (`_build_for_voltage`) | `solve_opts` deliberately omits `snes_error_if_not_converged=True` (the comment claims "the orchestrator handles non-convergence via checkpoint+rollback"). But Firedrake's default does **not** raise on SNES divergence — `solver.solve()` returns silently on a non-converged iterate. The `try/except Exception` in `run_ss` (line 256-259) only catches PETSc-level errors, not graceful SNES non-convergence. A divergent Newton step can be accepted as a successful time step, and a flat-but-wrong state can be declared "steady" by the plateau detector. Compare with `solvers.forsolve_bv` (line 69) which does `solve_opts.setdefault("snes_error_if_not_converged", True)`. | Sonnet ch.4 |
| 3 | **major** | `Forward/bv_solver/grid_per_voltage.py:228` (`_build_for_voltage`) and `Forward/bv_solver/solvers.py:68` (`forsolve_bv`) | `solve_opts = dict(params)` shallow-copies the full `solver_options` dict including the nested sub-dicts `bv_bc`, `bv_convergence`, `nondim`. Passing nested dicts as `solver_parameters` produces flattened keys like `bv_bc_reactions_0_k0` that PETSc does not recognize. Empirically the codebase works (PETSc warns rather than errors), but the noise hides real config issues and the dict mutation could break in newer PETSc versions. | Sonnet ch.4 |
| 4 | **major** | `Forward/bv_solver/config.py:24-26` (`_get_bv_cfg`) | When `alpha` is a list/tuple, `alpha_val=None` and the `(0,1]` range check is skipped entirely. Per-reaction validation (line 250-252) catches this for the multi-reaction path, but the legacy `_get_bv_cfg` path is unguarded. Not exercised by `plot_iv_curve_unified.py` (the multi-reaction path is taken), but a regression-trap if anything ever falls back to the legacy path. | Sonnet ch.3 |
| 5 | **major** | `Forward/bv_solver/config.py:259-260` (`_get_bv_reactions_cfg`) | `cathodic_species` and `anodic_species` indices are not range-checked against `[0, n_species)`, even though `cathodic_conc_factors.species` IS range-checked at line 239-243. A misconfigured reaction would only fail at UFL assembly with a confusing error. | Sonnet ch.3 |
| 6 | minor | `scripts/plot_iv_curve_unified.py:220` | CSV writer formats missing `z_factor` as the literal string `"nan"` rather than empty (`cd_mA_cm2`/`pc_mA_cm2` use `""` for missing). In normal operation all indices are populated; manifests only as a presentation inconsistency for failed voltages. | Sonnet ch.1 |
| 7 | minor | `Forward/params.py:90-100` | `__setitem__` bypasses `@dataclass(frozen=True)` via `object.__setattr__`. Production path uses `with_phi_applied`, but the escape hatch silently permits mutation of supposedly-frozen objects. Annotate or remove once legacy callers are migrated. | Sonnet ch.1 |
| 8 | minor | `Forward/bv_solver/forms_logc.py:349-357` | In the log-rate branch, the `else: anodic = Constant(0.0)` clause covers BOTH irreversible reactions and reversibles with `c_ref_model ≤ 1e-30`. The non-log-rate branch (line 375) has no such guard. Production R2 is irreversible so this is benign; a reversible reaction with `c_ref=0` would silently degrade to irreversible without a config-time error. | Sonnet ch.2 |
| 9 | minor | `Forward/bv_solver/forms_logc.py:448-473` | `ctx.update` omits diagnostic keys `_diag_E_eq_per_reaction`, `_diag_alpha_per_reaction`, `_diag_n_e_per_reaction` that `forms.py` exposes. Any downstream validation reading these from a logc context raises `KeyError`. Not in scope here, but a cross-formulation footgun. | Sonnet ch.2 |
| 10 | minor | `Forward/bv_solver/config.py:80-93` (`_default_bv_convergence_cfg`) | `conc_floor=1e-8` in defaults vs `1e-12` in `_make_bv_convergence_cfg` (the production factory). Callers that omit the key get a different value than the production preset uses. | Sonnet ch.3 |
| 11 | minor | `Forward/bv_solver/forms_logc.py:319` | `if E_eq_j_val is not None and E_eq_j_val != 0.0` is an exact-equality float comparison. Fine for production (E_eq is derived from a literal `0.0`), but fragile if a caller passes a tiny nonzero E_eq intended as "no correction". Use `abs(E_eq_j_val) > 1e-12`. | Sonnet ch.3 |
| 12 | question | `Forward/bv_solver/forms_logc.py:365-367` | Non-log-rate path uses Python `int` for `power` in UFL exponentiation; log-rate path wraps in `fd.Constant(float(...))`. Both work, but they may differ under pyadjoint annotation. Not exercised by this script (log-rate is on, and `adj.stop_annotating()` wraps the whole run). | Sonnet ch.2 |
| 13 | question | `Forward/bv_solver/dispatch.py:55-57` | If `SolverParams.solver_options` is `None` or non-dict, `_params_dict` returns `{}` and formulation silently defaults to `"concentration"` — could mask a misconfigured `SolverParams`. Add a warning. | Sonnet ch.2 |

### Items explicitly cleared

- **Sign conventions** on CD and PC (cathodic ORR is negative via the script's `scale=-I_SCALE`): correct.
- **Boltzmann residual sign** (`-z_scale·charge_rhs·z·c_bulk·exp(-z·φ)·w·dx`) matches the dynamic-species Poisson source convention. Anion accumulation near positive electrode is correctly captured.
- **z-ramp invariant** (`_set_z_factor` zeroes both `z_consts[i]` and `boltzmann_z_scale`): no other charge term escapes.
- **Log-rate ↔ non-log-rate equivalence**: algebraically identical (chunk 2 walked the derivation).
- **Nondim prefactors**: `electromigration_prefactor=1.0` exactly (since potential_scale=V_T); `poisson_coefficient = ε·V_T/(F·c·L²) = (λ_D/L)² ≈ 3.7e-8`; `charge_rhs_prefactor=1.0`. (Singularly perturbed problem — explains why the graded mesh + log-c primary variable matter.)
- **I_SCALE = 0.1833 mA/cm²** confirmed numerically with the standard physical constants and scales.
- **E_eq nondimensionalization**: `E_eq_model = E_eq_v / V_T` matches `eta = phi_applied - E_eq_model` in `_build_eta_clipped`.
- **Mesh marker chain**: `_make_bv_bc_cfg` 3/4/4 matches `make_graded_rectangle_mesh` (3=bottom electrode, 4=top bulk).
- **`SolverParams.with_phi_applied`** path is taken in the orchestrator's `_params_with_phi`; the list-fallback `_clone_params_with_phi` is unreachable on this script's path.
- **`adj.stop_annotating()`** cleanly suppresses the tape — no internal solver path calls `continue_annotating`.
- **Steric (Bikerman)** path is well-posed at the bulk: `Σ a_i c_i ≈ 0.012`, far above any floor.
- **Snapshot/restore for U and U_prev** is correct; the bug is specifically that `phi_applied_func` is not in `U.dat` and is missed by `_restore_U`.

---

## Agreement Analysis

Single-tier review (Level 1, Sonnet only). No cross-tier agreement to assess. **However**, both critical findings were verified by direct source inspection by the orchestrator before promotion (see `grid_per_voltage.py:101-108, 343-363, 224-238` in this conversation). The bisection bug is structural and unambiguous; the SNES bug depends on Firedrake-version behavior but is consistent with the explicit comment in the code that documents the omission of `snes_error_if_not_converged`.

If the criticals warrant higher confidence, re-run at Level 2 (adds Opus) on chunk 4 specifically — that's the only chunk with critical findings, and the rest of the codepath has only minor/question-level items.

---

## Suggested fix snippets (smallest diff)

**Bug 1 — `_march` bisection (`grid_per_voltage.py:343-363`):**

Track the previous-substep voltage explicitly and reset `paf` after `_restore_U`. One way:

```python
def _march(v0: float, v1: float, depth: int) -> bool:
    substeps = np.linspace(v0, v1, n_substeps_warm + 1)[1:]
    ckpt_outer = _snapshot_U(U)
    v_prev_substep = float(v0)
    for v_sub in substeps:
        ckpt_inner = _snapshot_U(U)
        paf.assign(float(v_sub))
        if run_ss(max_ss_steps_warm):
            v_prev_substep = float(v_sub)
            continue
        _restore_U(ckpt_inner, U, U_prev)
        paf.assign(v_prev_substep)              # <-- reset paf to last good
        if depth >= bisect_depth_warm:
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        v_mid = 0.5 * (v_prev_substep + float(v_sub))
        if not _march(v_prev_substep, v_mid, depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        if not _march(v_mid, float(v_sub), depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        v_prev_substep = float(v_sub)
    return True
```

**Bug 2 — SNES non-convergence (`grid_per_voltage.py:228-234`):**

```python
solve_opts = dict(params) if isinstance(params, dict) else {}
solve_opts.setdefault("snes_error_if_not_converged", True)
solver = fd.NonlinearVariationalSolver(
    problem, solver_parameters=solve_opts,
)
```

The orchestrator's existing `try/except Exception` in `run_ss` will then actually catch divergent solves. Remove the misleading comment.

**Bug 3 — solve_opts pollution (`grid_per_voltage.py:228` and `solvers.py:68`):**

Filter out the BV/nondim sub-dicts before passing as `solver_parameters`:

```python
NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
solve_opts = {k: v for k, v in (params or {}).items() if k not in NON_PETSC_KEYS}
```

Same change in `forsolve_bv`. Eliminates noisy "unknown PETSc option" warnings and is forward-compatible with stricter PETSc versions.

---

## Pointers

- Codepath map: `docs/plot_iv_curve_codepath.md`
- Per-chunk reports: `.verification/sonnet-chunk-{1..5}-report.md`
- Apr 27 production-rebuild writeup: `writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`
- Continuation-strategy rationale: `docs/CONTINUATION_STRATEGY_HANDOFF.md`
- Unified API surface: `docs/bv_solver_unified_api.md`
