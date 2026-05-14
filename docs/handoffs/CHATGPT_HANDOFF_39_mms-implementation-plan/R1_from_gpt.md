1. WHAT: Step 7 never asserts `formulation == "logc_muh"` or `ctx["logc_muh_transform"] is True`.
WHY: A factory regression to `logc` would not fail at the invariant layer; it would only surface later as confusing convergence failure or, worse, a misleading primary-variable comparison.
WHAT TO DO: Add hard asserts for `conv_cfg["formulation"] == "logc_muh"`, `ctx["logc_muh_transform"]`, and `ctx["mu_species"] == [h_idx]`.

2. WHAT: The stack invariants are too weak for the reaction schema.
WHY: `len(bv_reactions) == 2` does not prove this is R2e/R4e. Wrong `E_eq`, swapped reactions, missing H powers, wrong stoichiometry, or unscaled `K0_R4e` can pass.
WHAT TO DO: Assert exact reaction identities: `E_eq`, `alpha`, `n_electrons`, `reversible`, cathodic/anodic species, stoichiometry, H powers `2/4`, and `k0_R4e == K0_HAT_R4E * 1e-18`.

3. WHAT: The source builder reads static scaling values, but production residual uses live Firedrake coefficients for `phi_applied_func`, `stern_coeff_const`, `bv_k0_funcs`, `bv_alpha_funcs`, and `boltzmann_z_scale`.
WHY: Any continuation-style mutation between `build_forms` and source injection makes the manufactured source inconsistent with the actual residual.
WHAT TO DO: Either build source terms from the live ctx coefficients or assert live values equal the scaling values immediately before source construction.

4. WHAT: Clip/floor inactivity is assumed, not asserted.
WHY: `_build_bv_rates_ex` omits production eta clipping; closure helper may omit `phi_clamp`, `free_dyn_floor`, and `packing_floor`. That is only valid if every branch is provably inactive.
WHAT TO DO: Add runtime margin checks for `abs(eta_raw_j) < exponent_clip`, `abs(phi_ex) < min(phi_clamp_k)`, `abs(u_recon_i) < u_clamp`, `1 - A_dyn_ex > free_dyn_floor`, and `theta_inner_ex > packing_floor`.

5. WHAT: The planned “θ_inner discrete-min” check is not actually a discrete minimum.
WHY: `assemble(conditional(theta < threshold, 1, 0)*dx)` can miss pointwise or nodal violations depending on quadrature sampling. It proves little if the bad set is small.
WHAT TO DO: Interpolate/project `theta_inner_ex` to a suitable space and inspect `.dat.data_ro.min()`, and also keep a quadrature-point indicator check.

6. WHAT: `run_mms` return shape does not include `newton_converged`, `newton_iterations`, convergence reason, or residual norms, but Step 11 tests those.
WHY: The test plan is internally inconsistent; the convergence test cannot assert all solves converged unless the loop returns per-mesh solve status, including failures.
WHAT TO DO: Return a `per_mesh` diagnostics list or arrays for convergence flags, iteration counts, SNES reason, initial/final residual, and errors with `NaN` on failure. Do not silently drop failed meshes.

7. WHAT: Starting Newton from `U_manuf` can make the test pass as an interpolation test.
WHY: If the initial residual is already below tolerance, SNES may take zero iterations and all slopes can look perfect even with broken source terms.
WHAT TO DO: Add a pilot solve from a small deterministic perturbation of `U_manuf`, assert residual reduction and convergence back to the manufactured solution, and record initial residuals for the normal run.

8. WHAT: The risk table demands `||F_res||_final < snes_atol`, but the configured solve may converge by relative tolerance.
WHY: With large source magnitudes, final residual can legitimately exceed `1e-5`; conversely, absolute convergence alone does not prove discretization errors are solver-independent.
WHAT TO DO: Record SNES convergence reason and run a tolerance sweep at least on `N=32` and `N=64` to prove errors are insensitive to tighter `atol/rtol`.

9. WHAT: Quadrature sweep at only `N=32` is insufficient.
WHY: It can show a plateau at one mesh while quadrature error still contaminates the `N=64` convergence slope.
WHAT TO DO: Sweep degree on at least `N=32` and `N=64`, or run the full convergence chain at the candidate degree and one higher degree before pinning.

10. WHAT: No algebra-equivalence smoke test for the independently reimplemented shared-θ closure.
WHY: Step 4 is high-risk transcription work. Waiting for the full MMS pilot to catch a denominator/sign/packing mistake wastes time.
WHAT TO DO: As a pre-flight diagnostic only, compare independent `c_k_ex`, packing, and charge density against production `ctx["steric_boltzmann"]` after assigning `U_manuf`.

11. WHAT: Source independence is asserted by policy, not tested.
WHY: Accidentally using `ctx["ci_exprs"]`, production `steric_boltzmann`, or `U` in the source can self-cancel production mistakes and invalidate MMS.
WHAT TO DO: Add a source-form inspection test using UFL coefficient extraction, or a stricter code path, to verify source terms do not depend on `ctx["U"]` or `ctx["U_prev"]`.

12. WHAT: `TestMMSProductionGradedMesh` omits the acceptance criterion `newton_iterations < 30`.
WHY: Criterion 3 says fewer than 30 iterations; the planned test only checks convergence.
WHAT TO DO: Add `assert graded_mesh_results["newton_iterations"] < 30`.

13. WHAT: Graded-mesh thresholds are planned as one global L2/H1 threshold.
WHY: A loose global threshold can hide a single-field regression, especially for `mu_H` and `phi`.
WHAT TO DO: Pin per-field thresholds from pilot data, with separate entries for `u_O2`, `u_H2O2`, `mu_H`, and `phi`.

14. WHAT: The proton concentration diagnostic is missing.
WHY: Primary-variable convergence of `mu_H` is necessary, but users will compare against existing logc MMS concentration reporting. A reconstruction bug in reporting could go unnoticed.
WHAT TO DO: Report `c_H_L2` using `exp(U.sub(h_idx) - em*z_H*U.sub(phi_idx))`, not `exp(U.sub(h_idx))`. Keep assertions on primaries, but save concentration diagnostics.

15. WHAT: Factory verification claims it can inspect `bv_stern_capacitance_model` on the returned tuple.
WHY: That converted Stern coefficient is created during `build_forms` and stored in `ctx["nondim"]`, not directly by the factory.
WHAT TO DO: Make Step 2 verification build a context/forms once, then inspect `ctx["nondim"]["bv_stern_capacitance_model"]` and `ctx["stern_coeff_const"]`.

16. WHAT: The broken-config acceptance criterion is marked “optional but recommended.”
WHY: It is the only direct proof that the invariant layer catches factory drift. Optional means it will be skipped when time gets tight.
WHAT TO DO: Make `TestMMSAsserts` mandatory and parametrize broken configs: wrong formulation, no Stern, log-rate off, one counterion, wrong counterion identity, water on, cation hydrolysis on, small `dt`, loose SNES, wrong R4e factor, suppressed Poisson.

17. WHAT: Pytest-xdist/cache collision risk is acknowledged but not handled.
WHY: Fixed `/tmp/firedrake-tsfc`, `/tmp/pyop2`, and fixed `StudyResults/...` paths can collide across workers.
WHAT TO DO: Set worker-specific cache dirs before importing Firedrake when `PYTEST_XDIST_WORKER` exists, or mark these tests serial. Also make artifact writes atomic or worker-scoped.

18. WHAT: The test file may import Firedrake/MMS code too early.
WHY: `skip_without_firedrake` does not protect collection if the module import itself imports Firedrake and Firedrake is absent.
WHAT TO DO: Mirror the existing test’s lazy-import pattern: import `run_mms` and `verify_on_graded_production_mesh` inside fixtures/helpers only.

VERDICT: ISSUES_REMAIN