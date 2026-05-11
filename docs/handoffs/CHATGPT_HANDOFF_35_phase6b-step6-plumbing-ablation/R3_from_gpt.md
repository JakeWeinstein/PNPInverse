1. **WHAT:** A0b still does not test the actual residual wiring as written. You propose rebuilding `R_net_default_at_A0_state` and assembling `R_net` and `-R_net`.
**WHY:** That proves the algebraic expression is nonzero and antisymmetric. It does not prove those terms are actually present in `F_res`, nor that the H/K slots were wired correctly in the built form.
**WHAT TO DO:** Store the actual hydrolysis residual contribution terms on `ctx`, e.g. `_cation_hydrolysis_H_residual_term` and `_cation_hydrolysis_K_residual_term`, and assemble those. Or assemble the residual vector delta between hydrolysis-enabled and hydrolysis-disabled forms at the same state.

2. **WHAT:** `float(fd.assemble(R_net * v_h * ds))` is not a scalar flux if `v_h` is a mixed-space test function component. It assembles a linear form/vector, not a float.
**WHY:** The proposed A0b pseudo-code is not Firedrake-correct.
**WHAT TO DO:** For scalar flux diagnostics, assemble `fd.assemble(R_net * ds(marker))`. For residual-slot wiring, assemble the actual residual vector contribution and test the H/K block norms or sums deliberately.

3. **WHAT:** A0b’s antisymmetry check is tautological if you manually assemble `R_net` and `-R_net`.
**WHY:** `phys_h_flux + phys_k_flux = 0` then follows from the test code, not from solver wiring.
**WHAT TO DO:** The anti-symmetry check must read from the built H and K residual terms, not from two expressions constructed in the test.

4. **WHAT:** `|phys_h_flux| > 1e-30` is too weak.
**WHY:** A near-zero scale bug, missing λ, wrong nondim factor, or area-normalization mistake can pass. The expected A0 physical path is not merely nonzero; at λ=1 it should be consistent with `k_des * Γ`.
**WHAT TO DO:** Gate on magnitude: compare assembled average `R_net` against `k_des * gamma` or the existing mass-balance expression within a real tolerance, e.g. rel `5e-3`.

5. **WHAT:** The manufactured-aware diagnostic snippet reads `ctx["nondim"]["bv_convergence"]`, but current forms store convergence config at `ctx["bv_convergence"]`.
**WHY:** The collector will fail to detect manufactured runs and will continue emitting physical diagnostics.
**WHAT TO DO:** Use `ctx.get("bv_convergence", {})`.

6. **WHAT:** A3 gate 4, `Γ_A3 > Γ_A0`, is too weak to prove the override reached Picard.
**WHY:** If forms use the override but `update_gamma_from_solution` does not, solution changes could still perturb Γ slightly, or Γ could fail to move enough despite partial wiring.
**WHAT TO DO:** Compute `gamma_ss_langmuir(...)` from A3’s override-aware F0/c_H diagnostics and assert `gamma_A3` matches the predicted Γ within Picard tolerance.

7. **WHAT:** A3 gate 3 uses the wrong denominator concept. `amplification_from_singh_A0` is `⟨c_K*pka_factor⟩ / ⟨c_K⟩`, not necessarily `pka_factor_avg_A0`.
**WHY:** The proposed ratio can fail because of c_K/pKa spatial weighting, not because plumbing is wrong.
**WHAT TO DO:** For A3, assert `amplification_from_singh_A3 ≈ pka_factor_avg_A3` directly. If using a ratio to A0, divide by `amplification_from_singh_A0`, not `pka_factor_avg_A0`.

8. **WHAT:** Gates 1 and 2 do not prove “override-in-form” unless diagnostics assemble the exact pKa expression stored by forms.
**WHY:** If diagnostics rebuild the override expression correctly but forms use the old solved-σ expression, gates 1 and 2 still pass.
**WHAT TO DO:** Make `collect_v10a_rung_diagnostics` consume `ctx["_cation_hydrolysis_pka_shift_expr"]` from form-build, not rebuild its own independent expression.

9. **WHAT:** The structural test’s “only Constants” language is wrong. `r_H_El_pm_func` is a Firedrake `Function` coefficient, not a simple Constant.
**WHY:** A correct override pKa expression may still contain non-U coefficients.
**WHAT TO DO:** Test specifically that the expression does not depend on `ctx["U"]`. Allow bundle scalar coefficient Functions.

10. **WHAT:** `_cation_hydrolysis_sigma_S_active_expr` is ambiguous. In override mode it sounds like the fake pKa σ, but your test also wants the Stern σ that remains solved-field dependent.
**WHY:** This will cause tests to assert contradictory things about the same stored expression.
**WHAT TO DO:** Store two names: `_cation_hydrolysis_sigma_S_expr` for solved Stern σ, and `_cation_hydrolysis_pka_sigma_S_expr` for the pKa-side fake-or-solved σ.

11. **WHAT:** A1/A2 “positivity” is mostly tautological in log-concentration form.
**WHY:** `c = exp(u)` is nonnegative by construction. This does not catch severe depletion, near-zero K, NaNs, or unusable local boundary values.
**WHAT TO DO:** Use finite checks plus lower bounds, e.g. `c_K_boundary_avg > 0.01 * c_K_bulk` for A2 and no nonfinite assembled diagnostics.

12. **WHAT:** A0 reproduction at `rel=1e-9` is still stricter than the stated purpose needs.
**WHY:** Step 6 is not a bitwise regression campaign; it is a plumbing discriminator. A harmless PETSc variation can block the actual experiment.
**WHAT TO DO:** Keep the tiered table from R2: `<=1e-9 pass`, `1e-9..1e-6 rerun/document`, `>1e-6 block`. Do not make `1e-9` the only pass path unless you have repeat-run evidence.

VERDICT: ISSUES_REMAIN