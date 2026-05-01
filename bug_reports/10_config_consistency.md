# Bug Report: Config Consistency

**Focus:** Configuration propagation and default value consistency across modules
**Agent:** Config Consistency

---

## BUG 1: `packing_floor` and `softplus_regularization` config keys silently ignored
**File:** `Forward/bv_solver/config.py:50-56`, `Forward/bv_solver/forms.py:213,230`
**Severity:** MEDIUM
**Description:** `forms.py` consumes `conv_cfg["packing_floor"]` and `conv_cfg["softplus_regularization"]`, but `_get_bv_convergence_cfg` in `config.py` never emits these keys in its defaults dict. User settings for these keys are silently dropped, and the code falls back to whatever `dict.get()` default is used at the call site (which may differ from the intended default).
**Suggested fix:** Add `packing_floor` and `softplus_regularization` to the defaults dict in `_get_bv_convergence_cfg`.

## BUG 2: `blob_initial_condition` defaults differ between pipelines
**File:** `Inverse/inference_runner/config.py:83`, `FluxCurve/config.py:62`, `FluxCurve/bv_config.py:91`
**Severity:** MEDIUM
**Description:** `InferenceRequest` defaults `blob_initial_condition=True`, while `RobinFluxCurveInferenceRequest` and `BVFluxCurveInferenceRequest` default to `False`. The same solver parameters produce different initial conditions depending on which pipeline is used.
**Suggested fix:** Document the inconsistency or harmonize defaults.

## BUG 3: `evaluate_curve_loss_forward` lacks `observable_mode` parameter
**File:** `FluxCurve/curve_eval.py:117-144`
**Severity:** MEDIUM
**Description:** Relies on side-effect mutation of `steady.flux_observable` set far from callsite. Breaks when called independently from a different observable mode.
**Suggested fix:** Accept `observable_mode` as explicit parameter.

## BUG 4: `_normalize_kappa` hardcodes n_species=2 validation
**File:** `FluxCurve/run.py:36-45`
**Severity:** LOW
**Description:** Rejects valid multi-species configurations without helpful error message.

## BUG 5: RecoveryConfig defaults diverge between Inverse and FluxCurve pipelines
**File:** `Inverse/inference_runner/config.py:91-118`, `FluxCurve/config.py:10-30`
**Severity:** LOW
**Description:** `max_attempts` (15 vs 8), `anisotropy_only_attempts` (3 vs 1), `tolerance_relax_attempts` (1 vs 2) differ between pipelines. Same problem gets different recovery strategies.

## BUG 6: Surrogate training recovery config fragile dependency on parent defaults
**File:** `Surrogate/training.py:108-114`
**Severity:** LOW
**Description:** `max_attempts=4` exactly depends on parent's `tolerance_relax_attempts=2` default. Upstream changes silently break the budget.

## BUG 7: `_n_evals` counter inconsistent between FD and autograd paths
**File:** `Surrogate/objectives.py:197-231`
**Severity:** LOW
**Description:** FD path counts 9x more evals per optimizer step than autograd path. Makes eval metrics non-comparable across surrogate types.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 0     |
| MEDIUM   | 3     |
| LOW      | 4     |

**Most actionable:** Bug 1 -- config keys consumed by forms.py are not emitted by the config builder, causing silent drops of user settings.
