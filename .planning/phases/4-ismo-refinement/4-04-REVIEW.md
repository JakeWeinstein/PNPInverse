# 4-04 Plan Review: PDE Evaluation & Data Integration

**Verdict: FLAG**

Two issues require clarification or minor fixes before implementation, but none are blocking.

---

## Strengths

1. **Clear goal and well-scoped module.** The plan defines exactly what this module does (PDE eval, comparison, merge, quality) and explicitly defers decisions to the orchestrator (4-02) where appropriate (e.g., quality flag handling).

2. **Factory function for solver config is the right call.** Referencing the same constants from `scripts/_bv_common.py` at runtime eliminates config drift, which would be the single most dangerous failure mode in this pipeline. The frozen dataclass prevents accidental mutation.

3. **Correct delegation to existing infrastructure.** The plan reuses `generate_training_dataset` and `generate_training_dataset_parallel` from `Surrogate/training.py` rather than writing new PDE-solving code. This is exactly right -- those functions already handle interpolation of failed points, checkpointing, and warm-start chains.

4. **Immutable result dataclasses throughout.** Every return type is `frozen=True`, consistent with the codebase's pattern and the project's coding style rules.

5. **Versioned output paths with no in-place mutation of `training_data_merged.npz`.** This is critical for reproducibility and rollback.

6. **Duck-typed surrogate interface (`predict_batch`).** Verified: all 6 model families in the codebase (BVSurrogateModel, PODRBFSurrogateModel, NNSurrogateModel/EnsembleMeanWrapper, GPSurrogateModel, PCESurrogateModel) implement `predict_batch()` returning a dict with `current_density` and `peroxide_current` keys.

7. **Quality checks flag but don't auto-remove.** This is the correct separation of concerns -- the orchestrator decides tolerance policy.

---

## Issues

### Issue 1: `_build_voltage_grid` is private to `overnight_train_v11.py` -- not importable (Severity: MEDIUM)

The plan says `make_standard_pde_bundle` will call `_build_voltage_grid()` described as "the same function used by `overnight_train_v11.py`." However, `_build_voltage_grid` is a module-private function (underscore prefix) defined inside `scripts/surrogate/overnight_train_v11.py`, which is a script, not an importable module. The plan cannot simply import it.

**Fix:** Either (a) copy the voltage grid literal into `make_standard_pde_bundle` with a comment referencing the source, (b) extract it to a shared location like `Surrogate/constants.py` or `scripts/_bv_common.py`, or (c) load `phi_applied` from the existing `training_data_merged.npz` file (safest -- guarantees exact match with training data). Option (c) is recommended since it also validates the grid length at construction time.

### Issue 2: Existing `.npz` has extra keys (`n_existing`, `n_gapfill`) not mentioned in the plan (Severity: LOW)

The plan says `integrate_new_data` loads keys `parameters`, `current_density`, `peroxide_current`, `phi_applied`. The actual `training_data_merged.npz` also contains `n_existing` and `n_gapfill` metadata keys. These won't cause a crash (they'll just be dropped from the augmented output), but the provenance story is incomplete -- the original data's own provenance metadata will be lost.

**Fix:** When loading the existing `.npz`, carry forward any extra metadata keys to the output file. Or explicitly document that these keys are intentionally dropped in favor of the new provenance arrays.

### Issue 3: NRMSE normalization may differ from `validation.py` (Severity: MEDIUM)

The plan says `compare_surrogate_vs_pde` uses "the same normalization logic as `Surrogate/validation.py` (ptp-based with floor)." However, `validation.py` computes the floor as `max(global_ptp * 0.01, 1e-12)` where `global_ptp` is the range across the *entire test set*. In `compare_surrogate_vs_pde`, the "test set" is just the B_valid candidates from a single ISMO batch (potentially as few as 5-10 points). The global range computed from such a small, non-representative sample will be very different from the range across the full training/test set. This could produce NRMSE values that are not comparable to the validation metrics used elsewhere.

**Fix:** Either (a) pass the global range from the full training set as a parameter so the normalization denominator is consistent, or (b) use a fixed reference range derived from the existing training data (e.g., store `cd_ptp` and `pc_ptp` in the `PDESolverBundle` or `ISMOConfig`). This ensures the convergence threshold of 0.02 means the same thing regardless of batch composition.

### Issue 4: `B <= 100` safety bound may be too low for some strategies (Severity: LOW)

The plan validates `candidate_params` shape with `B <= 100`. The acquisition config in 4-02 defaults to `budget=30`, so this is fine for typical use. However, this is worth calling out as a magic number. If a future user changes the acquisition budget, the error message should explain why the bound exists (process spawn cost, PDE budget).

**Fix:** Make this configurable via `PDESolverBundle.max_batch_size` or at least include a clear error message referencing the PDE budget constraint.

---

## Missing Items

1. **No mention of `split_indices.npz` update.** The existing surrogate training pipeline uses `data/surrogate_models/split_indices.npz` to define train/test splits. When the dataset grows via ISMO, the split indices become invalid (they index into the original N=3491 array). The plan should specify how new points are assigned to train vs test. Recommendation: all ISMO-acquired points go to the training set (they were specifically chosen to improve the surrogate, not to serve as unbiased test data). The existing test indices remain valid if they index into the first `n_original` rows. This needs an explicit decision.

2. **No unit test specification.** Success criterion 8 mentions "unit tests verify data integration round-trip" but no test file is listed in "Files to Create." Add `tests/test_ismo_pde_eval.py` to the file list.

3. **No handling of the case where ALL candidates fail PDE evaluation.** The `PDEEvalResult` would have `n_valid=0` and empty arrays for `valid_params`, `current_density`, `peroxide_current`. The plan should specify that `integrate_new_data` handles this gracefully (no-op, returns the original dataset path) and that `compare_surrogate_vs_pde` raises or returns a sentinel when there's nothing to compare.

---

## Integration Risks

1. **`scripts/_bv_common.py` import path.** The `make_standard_pde_bundle` factory will need to import from `scripts/_bv_common`, which is in the `scripts/` directory (not a proper package). This may require `sys.path` manipulation or restructuring. Consider whether the needed constants (`I_SCALE`, `FOUR_SPECIES_CHARGED`, `SNES_OPTS_CHARGED`) should be moved to `Surrogate/constants.py` instead.

2. **4-03 vs 4-04 numbering confusion.** The 4-02 plan references "4-03 (Data Augmentation & Retraining)" while this plan is 4-04 but covers data augmentation. Meanwhile the actual 4-03 plan covers retraining (`ismo_retrain.py`). The dependency chain described in this plan (4-04) says it depends on 4-01 and 4-02, and that 4-05 consumes its output. But there are only plans 4-01 through 4-05. Verify the numbering is consistent across all plans -- the 4-02 plan's reference to "4-03 (Data Augmentation & Retraining)" appears to actually mean this plan (4-04). This is confusing but not blocking since the actual code interfaces are well-defined.

3. **Firedrake import discipline.** The plan correctly notes (success criterion 7) that Firedrake imports must be deferred. Since `make_standard_pde_bundle` calls `scripts/_bv_common.make_bv_solver_params`, verify that `_bv_common` does not trigger Firedrake imports at module level. If it does, the factory function itself needs to be lazy.
