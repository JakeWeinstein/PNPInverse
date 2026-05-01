# 4-03 Plan Review: Surrogate Retraining Pipeline

**Verdict: FLAG**

Solid overall design with well-reasoned design decisions, but several implementation-level issues need resolution before execution. None are architectural blockers; all are fixable within the existing plan structure.

---

## Strengths

1. **Design decisions are well-documented and correct.** The rationale for warm-start NN only, from-scratch for fast models, and ISMO data into training-only are all sound.

2. **Immutable pattern consistently applied.** All retraining functions return new model objects. This matches the project coding style and prevents partial-update corruption during failures.

3. **Quality check with fallback is a good safety net.** The warm-start -> degradation check -> from-scratch fallback is the right escalation pattern.

4. **ISMORetrainResult captures full provenance.** Having old/new error, ratio, method, and save path in a single dataclass enables the convergence monitor (4-05) to make informed decisions.

5. **Comprehensive test plan.** 14 test cases covering data merging edge cases, dispatch, quality checks, and per-model-type retraining with synthetic data.

---

## Issues

### Issue 1: Normalizer re-computation will cause catastrophic weight mismatch (Severity: HIGH)

The plan correctly identifies that normalizers must be re-computed on merged data. However, the warm-start approach (step 2e) loads old weights that were trained against old normalizers, then immediately uses them with new normalizers. If the data distribution shifts meaningfully (which ISMO guarantees -- it adds points in previously undersampled regions), the old weights will produce nonsensical outputs on the first forward pass with new normalizers, and the fine-tuning LR of 1e-4 may be too low to recover.

**Fix:** Add an explicit weight-adjustment step after re-computing normalizers. Specifically, the first and last layers can be analytically corrected for the normalizer shift: `W_new = W_old * (old_std / new_std)`, `b_new = b_old + W_old * (old_mean - new_mean) / new_std`. Alternatively, increase `nn_retrain_epochs` default from 100 to 500 and `nn_retrain_lr` from 1e-4 to 5e-4 to give the network more room to adapt. At minimum, the plan should acknowledge this risk and log the normalizer drift magnitude (e.g., `max(|new_mean - old_mean| / old_std)`) as a diagnostic.

### Issue 2: Duplicate detection in log-space vs linear-space ambiguity (Severity: MEDIUM)

The plan specifies `duplicate_param_tol: float = 1e-8` with L2 distance on raw parameter rows. However, parameters are stored in linear space: k0_1 ~ 1e-4, k0_2 ~ 1e-5, alpha ~ 0.5. A tolerance of 1e-8 would never detect duplicates for the alpha columns (values ~0.5) and would detect near-duplicates only for k0 values that are essentially identical to machine precision.

More critically, two parameter sets could have identical k0 values in log-space (which is the model's input space) but differ slightly in linear space beyond 1e-8, or vice versa. The duplicate check should operate in the same space the models use -- log-space for k0 columns, linear for alpha columns.

**Fix:** Transform k0 columns to log10 before computing L2 distance, or use a relative tolerance. A tolerance of 1e-6 in log-space would be more practical.

### Issue 3: `_finetune_nn_member()` duplicates the training loop (Severity: MEDIUM)

The plan acknowledges that calling `fit()` won't work because it re-initializes weights, and suggests writing `_finetune_nn_member()` with a custom training loop. This duplicates the ~50-line training loop from `NNSurrogateModel.fit()`, creating a maintenance burden. Any future changes to the training loop (loss function, scheduler, batch handling) would need to be mirrored.

**Fix:** Consider adding an optional `state_dict` parameter to `NNSurrogateModel.fit()` that, if provided, loads weights after model construction but before training. This is a 3-line change to `fit()` (insert after line 359 of nn_model.py) and eliminates the need for a separate training loop. The plan's `_finetune_nn_member()` approach works but should at minimum reference the original training loop's scheduler parameters (T_0=500 in fit() vs T_0=50 in the plan) to avoid silent divergence.

### Issue 4: `retrain_surrogate()` does data merging internally, but `retrain_surrogate_full()` also calls `save_merged_data()` (Severity: LOW)

Task 4 describes two public APIs: `retrain_surrogate()` (which calls `merge_training_data` and `update_split_indices` internally) and `retrain_surrogate_full()` (which wraps `retrain_surrogate()` and also calls `save_merged_data()`). This creates confusion about which function does the data merging. If a caller uses `retrain_surrogate()` directly, they get merged data returned implicitly (since the per-type functions receive `merged`), but the merged data is not saved and the updated split indices are not returned.

**Fix:** Either make `retrain_surrogate()` a thin dispatch-only function (taking pre-merged data) and have `retrain_surrogate_full()` do all the merging/saving, or remove one of the two public APIs. The current dual-API design invites misuse.

### Issue 5: Dependency metadata says `depends_on: [4-01, 4-02]` but 4-04 is listed in the prompt (Severity: LOW)

The YAML header says `depends_on: [4-01, 4-02]`. The plan does not actually depend on 4-04 (PDE evaluation) -- 4-03 consumes already-acquired data. This is correct. However, 4-04's plan says it depends on 4-03 for acquisition, which is actually about 4-02 (acquisition strategy). The cross-references between 4-03 and 4-04 are slightly muddled but not blocking.

### Issue 6: GPSurrogateModel has no `.config` attribute (Severity: LOW)

For POD-RBF, BVSurrogateModel, and PCE, the plan accesses `model.config` to pass to the new model constructor. However, `GPSurrogateModel.__init__()` takes only `device`, not a config object. The GP retraining function correctly handles this by just calling `GPSurrogateModel(device=device)`, but the `_check_retrain_quality()` helper and dispatch code should be aware that GP doesn't follow the config-copy pattern.

**Fix:** No code change needed, but the implementer should note this asymmetry.

### Issue 7: CosineAnnealingWarmRestarts T_0 mismatch (Severity: LOW)

The plan specifies `CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6)` for warm-start fine-tuning, but the original training in `nn_model.py` uses `T_0=500`. With only 100 fine-tuning epochs, T_0=50 gives exactly 2 restarts (at epoch 50 and epoch ~150, which won't be reached). This is probably fine, but should be explicitly noted as intentional rather than a copy error.

---

## Missing Items

1. **No rollback mechanism.** If retraining fails mid-way through an ensemble (e.g., member 3 of 5 crashes), the plan does not specify what happens. The already-retrained members 0-2 are saved to disk, but the function cannot return a valid ensemble. Add error handling: catch exceptions per-member, log failures, and either retry or return the best available ensemble.

2. **No disk space / path collision handling.** If `ismo_iter_1/` already exists from a previous interrupted run, `save_merged_data()` would silently overwrite. Add a check or use atomic writes (write to temp, then rename).

3. **No NNSurrogateModel in the model-loading integration test.** The plan's tests create small NNSurrogateModels from scratch but don't test the warm-start path through `NNSurrogateModel.load()` -> retrain -> `NNSurrogateModel.save()` round-trip. This is where normalizer serialization bugs would surface.

4. **`validate_surrogate()` type hint says `BVSurrogateModel` but accepts any surrogate.** The plan uses `validate_surrogate()` on all model types. This works in practice (duck typing), but the type annotation in `validation.py` line 16 could cause type-checker warnings. Not blocking, but worth a `# type: ignore` or updating the signature to `Any`.

---

## Integration Risks

1. **4-05 (convergence monitor) expects `ISMORetrainResult` but the exact field names must match.** If 4-05 is implemented first with stub types, ensure the field names in `ISMORetrainResult` here are treated as the contract. Recommend implementing 4-03 before 4-05.

2. **Model loading at ISMO iteration N > 0.** The plan saves retrained models to `ismo_iter_N/` paths, but the existing `load_nn_ensemble()` function in `ensemble.py` expects a fixed directory structure (`member_i/saved_model/`). The ISMO runner (4-05) will need to either use a modified loader or point `load_nn_ensemble()` at the iteration-specific directory. The plan's save path `nn_ensemble/ismo_iter_{iteration}/member_{i}/saved_model/` is compatible with `load_nn_ensemble()` if the base dir is set to `nn_ensemble/ismo_iter_{iteration}/`. This should work but is worth a unit test.

3. **GP retraining with growing N.** The plan correctly warns about O(N^3) scaling at N > 1500, but ISMO could push past this within a few iterations if the initial dataset is already ~500. The warning threshold should perhaps be lower (e.g., 1000) or include an estimated time.
