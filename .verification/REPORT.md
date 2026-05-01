# Verification Report — Round 2 (Post-Fix)

**Target:** Physics validation checks across 13 files (1 new + 12 modified)
**Date:** 2026-04-13
**Level:** 1 (Sonnet re-verification of 7 critical fixes)
**Agents:** Sonnet x4
**Verdict:** PASS — all 7 critical fixes confirmed correct. No new issues introduced.

## Fix Verification

| # | Fix | Verdict | Notes |
|---|-----|---------|-------|
| 1 | Config keys in `robust_forward.py` | PASS | Correct nested path, correct defaults. No remaining flat-key instances in codebase. |
| 2 | Config keys in `hybrid_forward.py` | PASS | Same fix, same confirmation. |
| 3 | Gradient/objective in `bv_curve_eval.py` | PASS | All 3 locations (single-obs, multi-obs primary, multi-obs secondary) correctly skip both. |
| 4 | Transient validation in `forward.py` + `__init__.py` | PASS | Mid-loop blocks fully removed. Post-convergence validation intact. Unused imports cleaned. |
| 5 | Interpolation in `training.py` | PASS | Direct magnitude check replaces vacuous validate_observables call. |
| 6 | Single-observable in `observables.py` | PASS | Honest F2-only check. Dead params (phi_applied, V_T) are harmless. |
| 7 | .npz keys in `compute_adjoint_gradients_v16.py` | PASS | Keys now match `_save_forward_checkpoint()` output. |

## Remaining Warnings (pre-existing, not regressions)

| # | Location | Issue | Severity |
|---|----------|-------|----------|
| 8 | `validation.py` | `exponent_clip` param unused; W1 never fires from `validate_solution_state` | low |
| 9 | `validation.py` | F1+F4 double-fire on negative concentrations | medium |
| 10 | `forms.py:346` | `E_eq != 0.0` guard should be `is not None` | medium |
| 11 | `bv_curve_eval.py` + `training.py` | I_lim=1.0 hardcoded vs 2.0 in overnight_train | medium |
| 12 | `hybrid_forward.py` | z=0 acceptance tightened (behavioral change) | low |
| 13 | `compute_adjoint_gradients_v16.py:270` | Wrong solver_params index for phi in log message | low |
| 14 | Subprocess workers | W-level warnings silently discarded | low |
