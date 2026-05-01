# Re-Verification Report: Chunk 4
**Scope**: Surrogate/training.py, scripts/surrogate/overnight_train_v16.py, scripts/surrogate/compute_adjoint_gradients_v16.py
**Date**: 2026-04-13

---

## Fix #5 — training.py `_interpolate_failed_points()` magnitude check

**Status: PARTIALLY CORRECT — threshold value inconsistency is a real problem**

The fix is in place at `training.py:250`:
```python
if abs(result[i]) > 1.0 * 1.05:  # F2: exceeds diffusion limit
```

The replacement is structurally correct: it removes the `validate_observables` call (which required a paired `pc` observable not available in single-flux context) and instead does a direct magnitude check for the F2 condition only.

**However, the threshold value `1.0` is wrong relative to the physics.**

In `overnight_train_v16.py`, I_LIM is computed correctly as:
```python
I_LIM = 2.0 * max(FOUR_SPECIES_CHARGED.c0_vals_hat)
# c0_vals_hat = [1.0, 0.0, 0.2, 0.2] -> max = 1.0 -> I_LIM = 2.0
```

So the physically correct diffusion limit is **2.0**, not **1.0**.

The check in `_interpolate_failed_points` uses `1.0 * 1.05 = 1.05`. The check in `overnight_train_v16.py`'s validation function uses `2.0 * 1.05 = 2.1`. These will produce different rejection behavior: interpolated values between 1.05 and 2.1 will be rejected in `_interpolate_failed_points` but accepted by the primary validation. This could discard valid interpolated data.

The same inconsistency exists at `training.py:187` where `validate_observables` is called with `I_lim=1.0` for convergence validation. Both occurrences use `I_lim=1.0` while the correct value (matching the physics in overnight_train_v16.py) is `2.0`.

**Verdict**: Fix #5 is structurally sound (direct check is correct approach) but uses the wrong threshold constant. Should be `2.0 * 1.05` not `1.0 * 1.05`.

---

## Fix #7 — compute_adjoint_gradients_v16.py `_load_forward_data()` key names

**Status: CORRECT — keys now match what overnight_train_v16.py saves**

The fix changed lookups from `"obs_cd"/"obs_pc"` to `"current_density"/"peroxide_current"`.

Confirmed in `overnight_train_v16.py`, the `_save_forward_checkpoint()` function at line 715 saves:
```python
np.savez(
    str(tmp_path),
    parameters=params,
    current_density=cd,
    peroxide_current=pc,
    converged=converged,
    timings=timings,
    phi_applied=phi_applied,
    n_completed=np.array([n_completed]),
)
```

The post-fix code in `compute_adjoint_gradients_v16.py:455-456` reads:
```python
obs_cd_all = data["current_density"] if "current_density" in data.files else None
obs_pc_all = data["peroxide_current"] if "peroxide_current" in data.files else None
```

The keys match exactly. The `in data.files` guard is defensive and correct. Fix #7 is fully correct.

---

## I_lim=1.0 vs I_LIM=2.0 Inconsistency (pre-existing warning)

**Status: STILL PRESENT — same inconsistency in multiple locations**

This was flagged as a warning, not critical. It remains unresolved:

- `Surrogate/training.py:187` — `validate_observables(..., I_lim=1.0, ...)` (wrong, should be 2.0)
- `Surrogate/training.py:250` — `abs(result[i]) > 1.0 * 1.05` (wrong, should be `2.0 * 1.05`)
- `scripts/surrogate/overnight_train_v16.py:408` — `I_LIM = 2.0 * max(...)` = 2.0 (correct)
- `scripts/surrogate/compute_adjoint_gradients_v16.py:465` — `validate_observables(..., I_lim=2.0, ...)` (correct)

The inconsistency means training.py will incorrectly reject valid samples with `|current_density|` between 1.05 and 2.1 (the actual physical limit). This is not a new issue but Fix #5's threshold compounds it.

---

## New Issues Introduced by the Fixes

**None** were introduced. Fix #7 is clean. Fix #5 does not introduce new logic paths; the structural approach is the right one for the single-flux context. The pre-existing threshold bug in `I_lim=1.0` was already there at `training.py:187` — Fix #5 merely used the same wrong constant.

---

## Summary

| Item | Status |
|---|---|
| Fix #5 structural correctness (no paired-observable dependency) | CORRECT |
| Fix #5 threshold value (1.0 * 1.05 vs correct 2.0 * 1.05) | BUG — pre-existing, compounded by fix |
| Fix #7 key name alignment with savez | CORRECT |
| I_lim=1.0 vs I_LIM=2.0 inconsistency | STILL PRESENT (training.py lines 187, 250) |
| New bugs introduced | NONE |

**Recommendation**: Change `training.py:187` `I_lim=1.0` to `I_lim=2.0` and `training.py:250` threshold from `1.0 * 1.05` to `2.0 * 1.05` to match the correct physics in overnight_train_v16.py.
