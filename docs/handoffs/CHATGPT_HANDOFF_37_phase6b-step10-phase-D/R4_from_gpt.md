1. WHAT: The A.2 lookup snippet is still wrong. The actual `per_k_hyd_records` entries contain `rungs`, not `lam_diagnostics`, in [phase_a2_v_kin.json](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json). WHY: the byte-equivalence check will raise `KeyError` before comparing physics. DO: select `target_record["rungs"]`, then the rung with `lambda_hydrolysis == 1.0`.

2. WHAT: The stated continuation order does not match `solve_grid_with_anchor`. That helper visits grid points by absolute distance from the anchor, not “all anodic then all cathodic” ([grid_per_voltage.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/grid_per_voltage.py:1038)). WHY: v4 says it is using existing topology but documents a different order. DO: either accept/document nearest-anchor ordering, or implement a custom visit order and test it.

3. WHAT: The per-Δβ solve topology omits the λ ramp. Existing v10a flow does λ=0 anchor/grid warm-walk, then λ ladder per V to λ=1. WHY: v4 budgets one Newton per V, but Phase D needs full physics `λ=1`; direct λ=1 from the warm walk is not stated or validated. DO: explicitly include the λ ladder per V, or add a convergence gate proving direct λ=1 resolves from the λ=0 snapshot.

4. WHAT: Because of #3, the wall budget is likely undercounted. If each V needs four λ rungs, per-eval PDE solves are closer to `anchor + 24 warm-walk + 24*4 λ-rungs`, not `~30`. WHY: 36 evals could be several thousand PDE solves, not ~1080. DO: budget using the actual continuation ladder the driver will run.

5. WHAT: The Stern pre-fit budget double-counts `Δ_β=0`. The Stern grid includes `0.0`, and the table also has a separate `Δ_β=0 baseline`. WHY: either runtime is overcounted, or the code wastes an expensive duplicate evaluation. DO: make Stern pre-fit “7 additional evaluations, reusing baseline” or remove `0.0` from the grid list and say baseline is injected.

6. WHAT: The first ablation grid point slightly violates its own safe-domain comment. Using rounded `6.43`, `-(15 - 6.43)/0.141` gives `T≈-15.00076`, not `-15`. WHY: a strict `|pka_shift_avg| <= 15` check can reject the grid endpoint. DO: compute from exact `baseline_T = beta_K_Cu * override_sigma` and put the lower target slightly inside, e.g. `-14.9`.

7. WHAT: The ablation grid formulas hard-code `6.43`. WHY: that duplicates derived state and already caused the small boundary miss. DO: compute every ablation grid value from `beta_K_Cu`, `override_sigma`, and exact target T values.

8. WHAT: `sigma_local_max_at_V_kin = abs(sigma_local_clamped_max)` is internally named wrong. WHY: the value is from the full V scan, not V_kin; a future implementer can use the wrong σ scale in `eps_beta` or `xatol_beta`. DO: rename to `sigma_local_clamped_max_over_grid`.

9. WHAT: `COMPARISON_FIELDS` still includes “fallback `gamma`” for `gamma_final`. WHY: fallback weakens the schema pinning that v4 says it wants. DO: assert exact `gamma_final` for the A.2 file; if future files rename it, that should be a schema migration, not silent fallback.

10. WHAT: The moved `SINGH_2016_CATION_PARAMS` plan needs a back-compat import test for object identity and mutation behavior. WHY: many tests/scripts import the table from `_bv_common`; “identical references” is promised but not enough unless tested before and after import-order permutations. DO: add tests importing `calibration.singh2016` first, `_bv_common` first, and both together, asserting `is` identity.

11. WHAT: The v4 plan still does not say how `beta_offset_pm2` is represented as a live coefficient. WHY: if it is baked into `cation_params`, runtime setters will not affect already-built forms; if it is a `Function`/`Constant`, diagnostics must read the same object. DO: specify `beta_offset_pm2_func` in the cation hydrolysis bundle, mirroring `r_H_El_pm_func`.

VERDICT: ISSUES_REMAIN