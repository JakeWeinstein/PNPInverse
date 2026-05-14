1. **NP electrode flux sign is internally inconsistent.**  
   WHAT: In §4.2 you define production `J_i = +D_i c_i (...)`, but then compute `J_i·n` as if `J_i,y = -D_i c_i ∂y(...)`. At `y=0`, production `J_i·n = -D_i c_i ∂y(...)`, not `+D_i c_i ∂y(...)`.  
   WHY: If implementation follows the manual expansion instead of `fd.dot(J_i, FacetNormal(mesh))`, the electrode source flips sign.  
   WHAT TO DO: Keep `g_i = dot(Jprod_i, n) - Σ_j s_ij R_j`; fix the hand derivation. If you want physical flux, define `N_i = -Jprod_i`.

2. **Your O₂ BV sign interpretation is wrong.**  
   WHAT: You say `−s_O2 R = +R` “sets `J_O2·n = +R`”. No. After IBP the boundary residual is `Jprod·n - sR`, so the natural condition is `Jprod·n = sR = -R` for O₂.  
   WHY: You are mixing production mobility flux with physical flux. O₂ consumption gives physical outward flux `N·n = +R`, but production `Jprod = -N`, so `Jprod·n = -R`.  
   WHAT TO DO: Rewrite §4.2/§4.4 sign commentary around `Jprod`, not physical flux.

3. **`U_prev = U_manuf` does not make the time term exactly zero in the discrete convergence test.**  
   WHAT: `U_prev` is a CG1 interpolant of the manufactured field. During the solve, `c(U_h) - c(U_prev_h)` is not identically zero.  
   WHY: With `dt=1e15` this is negligible, but the derivation’s “exactly 0” claim is false and can hide a low-order perturbation if someone changes `dt`.  
   WHAT TO DO: State it as “negligible by construction”; assert `dt_model` is huge, or add an explicit time source if you want exact cancellation.

4. **The muh implementation notes are unsafe if reused verbatim.**  
   WHAT: “keep `run_mms` outer-loop verbatim” is wrong for H. Existing logc code initializes/errors H as `u_H = ln c_H`; new code must initialize/error `U.sub(h)` against `μ_H^ex`. H concentration diagnostics must use `exp(μ_H - em z φ)`, not `exp(U.sub(h))`.  
   WHY: You can get bogus H errors or miss the exact reconstruction bug the test is supposed to catch.  
   WHAT TO DO: Add explicit μ-aware interpolation and error code.

5. **Your η safety ranges are numerically wrong.**  
   WHAT: For `|φ(x,0)| ≤ 2`, `η_R2e = 21.41 - φ - 27.04` is about `[-7.64, -3.64]`, not `[-30.4, -5.6]`. `η_R4e` is about `[-28.46, -24.46]`, not `[-51.3, -26.5]`.  
   WHY: The clip-safety proof is using bad arithmetic. The conclusion is still safe for the recommended parameters, but the evidence is wrong.  
   WHAT TO DO: Recompute and write the actual ranges for the chosen `α0, α1, γ`.

6. **The packing-floor proof overclaims.**  
   WHAT: §5.4 says `θ_inner^ex > 0.95` over the broad `|φ|≤2` envelope. With SO₄²⁻, `exp(2φ)` at `φ=2` makes the sulfate packing contribution about `0.19`, so `θ_inner` can be closer to `0.8`.  
   WHY: Still far above `1e-8`, but the stated lower bound is false.  
   WHAT TO DO: Prove floor inactivity for the actual recommended parameters, or sample/interpolate `min(theta_inner)` in the test and assert margin.

7. **Inactive clamp/floor coverage is overstated.**  
   WHAT: You claim clamp/floor-induced model errors would show up, then later say clip/floor behavior is deferred. The latter is correct.  
   WHY: Since `u_clamp`, `phi_clamp`, `exponent_clip`, `free_dyn_floor`, and `packing_floor` are all inactive at `u_exact`, wrong branch behavior is mostly untested.  
   WHAT TO DO: Move these to “guarded inactive assumptions,” not “covered behavior.”

8. **The exact stack assumptions need hard asserts.**  
   WHAT: The derivation assumes `use_reactions=True`, `bv_log_rate=True`, `use_stern=True`, `suppress_poisson_source=False`, water/cation hydrolysis disabled, no Γ slot, exactly 3 dynamic species, and two bikerman counterions.  
   WHY: A config drift can silently make the source builder cancel the wrong operator.  
   WHAT TO DO: Add explicit asserts in the MMS factory/source builder before injecting sources.

9. **The R4e strength is not pinned.**  
   WHAT: The demo sweeps `K0_R4e_factor`; the derivation never says which factor the MMS uses.  
   WHY: If the factor is tiny, R4e becomes numerically irrelevant and does not meaningfully test the irreversible 4e branch. If it is 1, it tests the branch but may stress Newton.  
   WHAT TO DO: Choose and document the factor, preferably with a lower-bound check on assembled `R_R4e` magnitude.

10. **`bv_c_ref_model_vals` is not exercised.**  
   WHAT: In the reaction path, production does not read legacy `scaling["bv_c_ref_model_vals"]`. Also `c_ref_model` is unused for R2e because it has an anodic species, and unused for R4e because it is irreversible.  
   WHY: Bugs in legacy c-ref scaling or reversible c-ref-anchored anodic logic will not be caught.  
   WHAT TO DO: Mark this explicitly as uncovered, or add a separate reversible anchored reaction MMS.

11. **Live continuation parameters can desynchronize the source.**  
   WHAT: Production uses live `phi_applied_func`, `stern_coeff_const`, `bv_k0_funcs`, `bv_alpha_funcs`, and `boltzmann_z_scale`. The derivation often says “read scaling.”  
   WHY: If any setter runs after form build, metadata and UFL constants/functions may diverge from a source built from stale values.  
   WHAT TO DO: Either assert no continuation setters are used in the MMS, or build sources from the same live parameter objects.

12. **Quadrature degree 8 is asserted, not demonstrated.**  
   WHAT: The source integrands include `exp`, `ln`, `cos`, and rational Bikerman terms. `degree=8` may be fine, but the derivation gives no evidence.  
   WHY: Quadrature error can fake a convergence-rate failure or flatten rates on fine meshes.  
   WHAT TO DO: Add a one-time quadrature sweep, e.g. degree 8/12/16, and pin the degree based on observed rate stability.

VERDICT: ISSUES_REMAIN