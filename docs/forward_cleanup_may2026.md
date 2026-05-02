# Forward-tree cleanup — May 2026

**Date:** 2026-05-02
**Scope:** Removed legacy 4sp/concentration backend + all dead-experiment
modules from the forward stack. Deleted v13/v15-era inference tests.
Verified bit-exact numerical equivalence of the production driver
against the pre-cleanup baseline.

## Recovery tag

| Tag | Commit | Description |
|---|---|---|
| `pre-cleanup-2026-05-02` | `6a46d98` | Production-stack snapshot. Contains **both** the new 3sp + Boltzmann + log-c + log-rate stack **and** the legacy 4sp/concentration backend that this cleanup removed. Use this if you need to inspect or run any deleted file. |

```bash
# Inspect any deleted file at the pre-cleanup state:
git show pre-cleanup-2026-05-02:Forward/bv_solver/forms.py
git show pre-cleanup-2026-05-02:scripts/studies/v24_3sp_logc_vs_4sp_validation.py

# Full revert to pre-cleanup tree (detached HEAD):
git checkout pre-cleanup-2026-05-02

# Selective revert of one phase:
git revert <phase-commit-hash>
```

## Phase commits

Each phase is a single commit (or small cluster) on top of the
production-stack snapshot. The cleanup phases run from `ebea17f`
through `4ede992`; `629666e` is the doc refresh that followed.

| Phase | Commit | Description | Net LOC |
|---:|:---|:---|---:|
| (prep) | `6a46d98` | snapshot: production forward stack | (n/a) |
| 1 | `ebea17f` | tag baseline + remove broken `v24` script | −725 |
| 2 | `92ef08a` | delete 7 dead `bv_solver/` experiments | −2,470 |
| 3 | `e72163d` | remove 4sp/concentration backend + prune `validation.py` | −1,338 |
| 4a | `3586ad0` | prune `solvers.py` + finalize `bv_solver/__init__.py` | −682 |
| 4d | `1ef4bcf` | prune `scripts/_bv_common.py` to production surface | −143 |
| 4e | `26737bf` | prune `Forward/params.py` (drop unused helpers) | −12 |
| 5 | `4ede992` | delete dead tests + migrate MMS test to log-c | −4,098 |
| (docs) | `629666e` | refresh `plot_iv_curve_codepath.md` | (docs) |
| **Total** | | | **−9,468** |

## What was removed

### Files deleted

#### `Forward/bv_solver/` (Phases 2 + 3)

| File | LOC | Reason |
|:---|---:|:---|
| `forms.py` | 548 | 4sp/concentration weak forms (Phase 3) |
| `grid_charge_continuation.py` | 632 | Strategy-B orchestrator (Phase 3); converged 3/13 at production resolution per V26 |
| `forms_log.py` | 443 | abandoned log-rate-only forms (Phase 2) |
| `forms_mixed_logc.py` | 345 | abandoned mixed-formulation experiment (Phase 2) |
| `gummel_solver.py` | 392 | abandoned operator-split solver (Phase 2) |
| `hybrid_forward.py` | 270 | abandoned z-ramp/FUS hybrid (Phase 2) |
| `robust_forward.py` | 703 | abandoned robust curve solver (Phase 2) |
| `stabilized_forward.py` | 206 | abandoned PSPG stabilized forward (Phase 2) |
| `stabilization.py` | 111 | helper for `stabilized_forward` (Phase 2) |

#### Tests + verification (Phase 5)

| File | LOC | Reason |
|:---|---:|:---|
| `tests/test_v13_verification.py` | 438 | v13 surrogate inference; entire pipeline removed |
| `tests/test_pipeline_reproducibility.py` | 581 | v13 PIP reproducibility |
| `tests/test_inverse_verification.py` | 1,155 | used `make_recovery_config` + `FOUR_SPECIES_CHARGED` (both removed) |
| `tests/test_surrogate_fidelity.py` | 555 | v15 surrogate fidelity gates |
| `tests/test_bv_forward.py` | 95 | wrapper for the verification script below |
| `scripts/verification/test_bv_forward.py` | 561 | strategies A–F all on concentration / 4sp; replaced by MMS coverage |
| `scripts/verification/mms_bv_4species.py` | 808 | 4sp MMS; superseded by `mms_bv_3sp_logc_boltzmann.py` |

#### Studies (Phase 1)

| File | LOC | Reason |
|:---|---:|:---|
| `scripts/studies/v24_3sp_logc_vs_4sp_validation.py` | 725 | broken on this tree; historical results remain at `StudyResults/v24_3sp_logc_vs_4sp_validation/` |

### Functions / symbols pruned (live files)

#### `Forward/bv_solver/solvers.py` (Phase 4a, 635 → 21 LOC)

Removed: `forsolve_bv`, `solve_bv_with_continuation`,
`solve_bv_with_ptc`, `solve_bv_with_charge_continuation`. Kept:
`_clone_params_with_phi` (used by `grid_per_voltage.py:185`).

#### `Forward/bv_solver/dispatch.py` (Phase 3, 125 → 51 LOC)

Removed: `_get_formulation`, `_params_dict`, the concentration-backend
import block, and the per-call routing logic. The three public
functions (`build_context`, `build_forms`, `set_initial_conditions`)
are now one-line wrappers that delegate directly to
`forms_logc.{build_context_logc, build_forms_logc, set_initial_conditions_logc}`.
`set_initial_conditions` still accepts `blob=False` for
backward-compat kwargs (silently ignored — blob ICs were a
concentration-formulation feature).

#### `Forward/bv_solver/__init__.py` (Phase 3 + 4a, 149 → 97 LOC)

Removed re-exports of the four `solvers.py` continuation helpers,
and the `grid_charge_continuation` re-exports
(`solve_grid_with_charge_continuation`, `GridChargeContinuationResult`,
`GridPointResult`). Module docstring rewritten to describe only the
production stack.

#### `Forward/bv_solver/validation.py` (Phase 3, 314 → 238 LOC)

Removed: `validate_steady_state` (no callers),
`check_clip_saturation` (only `overnight_train_v16.py` caller, which
was already broken by the `forms.py` deletion). Kept:
`ValidationResult` (lazy-imported by `observables.py:100`),
`validate_solution_state` (FluxCurve surrogate fast-path —
**deviation from the original plan**, which slated it for deletion),
`validate_observables`, `compute_i_lim_from_params`.

#### `scripts/_bv_common.py` (Phase 4d, 635 → 492 LOC)

Removed:

- Neutral-block constants: `D_O2_NEUTRAL`, `D_REF_NEUTRAL`,
  `K_SCALE_NEUTRAL`, `D_O2_HAT_NEUTRAL`, `D_H2O2_HAT_NEUTRAL`,
  `K0_HAT_R1_NEUTRAL`, `K0_HAT_R2_NEUTRAL`, `I_SCALE_NEUTRAL`.
- Species presets: `TWO_SPECIES_NEUTRAL`, `FOUR_SPECIES_CHARGED`.
- SNES presets: `SNES_OPTS_STRICT`.
- Helpers: `setup_pnpinverse_env`, `make_recovery_config`,
  `print_params_summary`, `print_redimensionalized_results`.

`make_bv_solver_params` keeps its full kwargs surface; the `species`
default flips from `FOUR_SPECIES_CHARGED` to
`THREE_SPECIES_LOGC_BOLTZMANN`, and the `formulation` default flips
from `"concentration"` to `"logc"` to match the production
dispatcher.

#### `Forward/params.py` (Phase 4e, 212 → 200 LOC)

Removed `with_phi0`, `with_a_vals`, `with_z_vals` (zero callers).

## What was kept (plan deviations)

The original plan would have removed these, but live-caller analysis
showed they're still in active use:

| Symbol | Lives in | Live caller(s) |
|:---|:---|:---|
| `Forward/bv_solver/sweep_order.py` | full file | `FluxCurve/bv_point_solve/predictor.py` re-exports `_apply_predictor` and `_build_sweep_order`; both are used in `FluxCurve/bv_point_solve/__init__.py` lines 236 / 520. |
| `validate_solution_state` | `validation.py` | `FluxCurve/bv_point_solve/__init__.py:715` and `FluxCurve/bv_point_solve/forward.py:310` (surrogate fast-path validation). |
| `with_c0_vals` | `Forward/params.py:194` | `FluxCurve/bv_curve_eval.py:499`. |
| `SNES_OPTS_CHARGED` | `_bv_common.py:182` | `scripts/plot_iv_curve_unified.py` (production driver), `scripts/studies/v18_logc_lsq_inverse.py` (v18 inverse template), `Surrogate/ismo_pde_eval.py`. Despite the "charged" name, this is the SNES preset the **logc** stack uses. |

The plan also specified migrating `tests/test_bv_forward.py` to log-c.
Instead I deleted it (along with `scripts/verification/test_bv_forward.py`)
because (a) the strategies A–D it ran exercised the concentration
backend that no longer exists, and (b) MMS convergence + the
production driver bit-exact baseline already cover the correctness +
robustness ground that file did. Net cost-benefit favored deletion.

## What was broken by intent

These callers depended on symbols this cleanup removed. They were
already on the path to being rebuilt against the new stack and are
not on the protected set; they fail with `ImportError` post-cleanup
and will need to be re-implemented on the production logc backend
(or recovered from the tag) when their workflows are revived:

- `scripts/surrogate/overnight_train_v16.py` — imports `forms.py`
  (deleted in Phase 3).
- `Surrogate/ismo_pde_eval.py:make_standard_pde_bundle` —
  references `FOUR_SPECIES_CHARGED` (deleted in Phase 4d). The
  module-level imports are deferred, so `import Surrogate` still
  works; only this function fails when called. Transitively breaks
  `scripts/surrogate/run_ismo.py`.
- `scripts/Inference/v15`–`v18*.py` — most import `robust_forward`,
  `stabilization`, or `stabilized_forward` (all deleted in Phase 2),
  or `solve_bv_with_charge_continuation` (deleted in Phase 4a).
- `scripts/Inference/Infer_BVMaster_charged_v15.py` — imports
  `solve_bv_with_charge_continuation`.
- Many `scripts/studies/test_*.py` and `scripts/studies/v18_test_*.py`
  files — the dead-experiment study scripts that exercised the
  modules deleted in Phase 2.

The decision is consistent with the user's stated goal: _"forward
architecture for review, then later build inverse scripts."_ The
inverse rebuild will use the cleaned-up forward surface
(`Forward.bv_solver` + `forms_logc` + `grid_per_voltage`), patterned
on `scripts/studies/v18_logc_lsq_inverse.py` which is preserved.

## Verification

| Check | Result |
|:---|:---|
| AST parse on every changed `.py` file | clean |
| `pytest --collect-only` | 325 tests collected, no import errors |
| `pytest -m "not slow"` | 315 pass; 6 fail in `test_autograd_gradient.py` + `test_multistart.py` — confirmed pre-existing by re-running on `pre-cleanup-2026-05-02` |
| Smoke-import production stack | `Forward.bv_solver`, `scripts._bv_common`, `scripts.plot_iv_curve_unified`, `scripts.studies.v18_logc_lsq_inverse`, `FluxCurve.bv_point_solve`, `Surrogate` all import cleanly |
| `scripts/plot_iv_curve_unified.py` at `MESH_NY=200`, 13 voltages | **bit-exact** match with pre-cleanup baseline at every voltage. Max relative CD error: 0.0. Max relative PC error: 0.0. Methods identical. 13/13 converged. Wall time 133.5s vs baseline 134.5s. |

The bit-exact comparison is the strongest possible numerical
equivalence for this stack — far better than the plan's tolerance of
"~0.01%". The cleanup did not change a single bit of the production
forward observable output.

## File-size after the cleanup

`Forward/bv_solver/` totals 3,197 LOC across 14 files (down from
~6,857 LOC across 21 files). All live files are under 600 LOC; most
are under 300.

| File | LOC |
|:---|---:|
| `solvers.py` | 21 |
| `dispatch.py` | 51 |
| `mesh.py` | 65 |
| `__init__.py` | 97 |
| `observables.py` | 115 |
| `boltzmann.py` | 129 |
| `sweep_order.py` | 163 |
| `nondim.py` | 168 |
| `validation.py` | 238 |
| `config.py` | 268 |
| `forms_logc.py` | 531 |
| `grid_per_voltage.py` | 545 |

## Pointers

- Cleanup plan that was executed: `~/.claude/plans/make-a-plan-to-expressive-milner.md`.
- Production stack API: `docs/bv_solver_unified_api.md`.
- Continuation strategy rationale: `docs/CONTINUATION_STRATEGY_HANDOFF.md`.
- Production driver codepath: `docs/plot_iv_curve_codepath.md` (refreshed for this cleanup).
- Project-conventions / hard rules: `CLAUDE.md`.
