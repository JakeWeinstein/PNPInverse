1. **WHAT:** A3 still does not prove the override reached the actual `R_net` used in `F_res`.
**WHY:** Gates 1-3 can pass if forms store the override-aware `pka_shift_expr`, diagnostics consume it, and Picard consumes it, but the actual residual `R_net` was accidentally built from the old solved-σ pKa expression. That is exactly the kind of split-brain bug this ablation is supposed to catch.
**WHAT TO DO:** Apply the A0b scalar residual check to A3 too: assemble the actual stored `_cation_hydrolysis_R_net_scalar_form` at A3 λ=1 and verify `∫R_net ds / area ≈ k_des * Γ_A3` within the mass-balance tolerance. Better: also store `_cation_hydrolysis_R_net_expr` and ensure all scalar/residual forms are built from that same object.

2. **WHAT:** “Gates 1+2 prove override-in-form” is still overstated.
**WHY:** They prove diagnostics read the ctx-stored pKa expression. They do not prove that expression was the one used inside the residual unless the stored residual/R_net artifacts are also checked.
**WHAT TO DO:** Reword: gates 1+2 prove override-in-stored-pKa/diagnostics. Add the A3 residual scalar gate above to prove override-in-residual.

3. **WHAT:** The manufactured diagnostics early-return still lacks an explicit top-level `c_K_boundary_avg`.
**WHY:** Current physical diagnostics put K in `F0_decomposition.c_K_avg`. Your manufactured path sets `F0_decomposition = None`, so A2 has no K observable unless you add one.
**WHAT TO DO:** Before returning for manufactured runs, always emit top-level `c_H_boundary_avg` and `c_K_boundary_avg` assembled directly from `ci[h_idx]` and `ci[counterion_idx]`.

4. **WHAT:** “All diagnostics finite” is underspecified and will conflict with intentional `None` fields.
**WHY:** Manufactured records deliberately set physical fields to `None`; JSON records also contain strings, bools, lists, and dicts.
**WHAT TO DO:** Define this as “all required numeric diagnostics for that ablation are finite,” with an explicit required-key list per ablation.

5. **WHAT:** The A0b vector-slot plan depends on Firedrake Cofunction splitting details that are not confirmed.
**WHY:** `fd.assemble(linear_form)` may not expose `.subfunctions[...]` exactly like a `Function` in all Firedrake versions.
**WHAT TO DO:** Add a small implementation precheck or test fixture proving how to split the assembled dual vector. If `.subfunctions` is unavailable, use mixed-space dof maps/block extraction explicitly.

6. **WHAT:** Summing residual DOF entries is a fragile proxy for integrated anti-symmetry.
**WHY:** It relies on matching H/K scalar spaces and basis partition behavior. It is probably fine here, but it is not the cleanest invariant.
**WHAT TO DO:** Keep the slot-presence/off-slot tests, but make the sign gate primarily compare scalar assembled forms: `assemble(H_flux_scalar) + assemble(K_flux_scalar)`, where those scalar forms are stored from the same canonical `R_net_expr`. Use DOF sums only as a secondary structural check.

7. **WHAT:** The A3 pseudo-code uses `bundle.k_hyd_func.values()[0]`.
**WHY:** That is not the pattern used elsewhere for R-space Firedrake Functions and may not work.
**WHAT TO DO:** Use `float(bundle.k_hyd_func)`, same as existing cation-hydrolysis diagnostics.

VERDICT: ISSUES_REMAIN