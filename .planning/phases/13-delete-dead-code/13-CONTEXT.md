# Phase 13: Delete Dead Code - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove old scripts and tests from prior pipeline iterations (v1-v12), keeping only v13 pipeline essentials. This is deletion only — no refactoring or code changes to kept files.

</domain>

<decisions>
## Implementation Decisions

### Deletion strategy
- Git-delete all dead code (no archiving — git history preserves everything)
- Single commit for all deletions: "chore(13): delete dead code"
- Contrast with Phase 12 which archived results to archive/ — scripts don't need that treatment

### Inference scripts (SCRP-01)
- Delete ALL 42 files in scripts/inference/ — none are v13-relevant
- The entire directory becomes empty after deletion

### Surrogate scripts (SCRP-02)
- Delete old versioned scripts: v8, v8_1, v9, v10, v11, v12 variants
- Delete non-v13 scripts: bcd_inference.py, cascade_inference.py, cascade_pde_hybrid.py, run_nopde_batch.py, run_v9_pipeline.sh, sweep_secondary_weight.py
- Delete Infer_PDE_only_v14.py (SCRP-05)
- Keep v13 pipeline: Infer_BVMaster_charged_v13_ultimate.py
- Keep utilities: build_surrogate.py, generate_training_data.py, train_nn_surrogate.py, validate_surrogate.py, multistart_inference.py
- Keep training scripts: train_improved_surrogate.py, overnight_train_v11.py

### Studies scripts (SCRP-03)
- Delete ~15 old scripts: all benchmarks (benchmark_*), Robin-era scripts (Generate_Robin*, Probe_RobinFlux*, Test_RobinFlux*), early BV studies (bv_k0_noise_sensitivity, plot_bv_k0_inference_results), early exploration (bfgs_lbfgsb_diffusion_failure_study, forward_solver_D_stability_study, forward_solver_feasibility_study, optimization_method_study), voltage range studies (extended_voltage_range_study, charged_voltage_range_study)
- Keep 4 recent scripts: run_multi_seed_v13.py, profile_likelihood_study.py, profile_likelihood_pde.py, sensitivity_visualization.py
- Delete scripts/generate_presentation_plots.py (top-level)

### BV scripts (SCRP-04)
- Delete bv_iv_curve.py and bv_iv_curve_symmetric.py (per requirements)
- bv_iv_curve_charged.py: check imports before deciding — delete if nothing references it
- scripts/_bv_common.py: check imports before deciding — delete if only used by deleted scripts

### Test files (TEST-01)
- Delete 10 test files: test_v11_e2e_pde.py, test_v11_surrogate_pde.py, test_bcd.py, test_cascade.py, test_cascade_pde_hybrid.py, test_ensemble_and_v12.py, test_inference_robustness.py, test_weight_sweep.py, test_nondim_audit.py, test_fixed_pde.py
- Note: test_fixed_pde.py added beyond original TEST-01 requirements (v10 fixed-PDE approach is dead)

### Claude's Discretion
- Whether to remove empty scripts/inference/ directory or leave it
- How to handle __pycache__ directories in deleted script folders
- Order of deletions within the single commit

</decisions>

<specifics>
## Specific Ideas

- User wants import-checking for borderline files (_bv_common.py, bv_iv_curve_charged.py) before deciding their fate
- overnight_train_v11.py kept despite "v11" in name — user considers it still useful for training

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- scripts/verification/ directory: V&V framework scripts — untouched by this phase
- tests/ V&V tests (test_mms_convergence, test_surrogate_fidelity, etc.): kept intact

### Established Patterns
- Phase 12 used git rm for deletions — same approach applies here
- Single atomic commit preferred over category-by-category

### Integration Points
- scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py may import from kept utilities — verify no broken imports after deletion
- tests/conftest.py may reference deleted test infrastructure — verify after deletion
- Phase 14 will verify all remaining imports and test suite

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-delete-dead-code*
*Context gathered: 2026-03-13*
