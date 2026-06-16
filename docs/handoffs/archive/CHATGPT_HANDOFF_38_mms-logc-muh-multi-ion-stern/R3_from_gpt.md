1. **Point 9 / point 4: R4e is not “competing”; it catastrophically dominates.**  
   WHAT: At `V_RHE = 0.55`, with `K0_R4e_factor = 1`, the R4e cathodic exponent is roughly `2*26.9 ≈ 54`, while R2e is roughly `1.254*6.1 ≈ 7.7`. Since `k0_R4e ≈ k0_R2e`, `R4e/R2e ~ exp(46)`, i.e. around `1e20`, not “at least 10%.”  
   WHY: Your `R4e_norm > 0.1*R2e_norm` guard is vacuous. More importantly, residual/source magnitudes can be enormous; `snes_rtol=1e-7` may allow a final residual that is still huge in absolute terms. This can wreck the convergence measurement or make it solver-tolerance-limited.  
   WHAT TO DO: Add an upper-bound guard too, e.g. require `R4e_norm/R2e_norm` in a documented finite window, or choose a smaller primary `K0_R4e_factor` for the convergence sweep. If you insist on factor `1`, prove with an actual pilot solve that Newton tolerance and quadrature error do not dominate the FE error.

2. **Point 6: the optional Stern-off cross-check is physically/numerically broken with Bikerman SO₄²⁻.**  
   WHAT: `φ^ex_NoStern(x,0) = φ_app ≈ 21.4`. Even though this is below `phi_clamp=50`, SO₄²⁻ has `q = exp(2φ) ≈ exp(42.8)`, so the shared-θ closure saturates: `P_SO4 → 1 - A_dyn`, hence `θ_inner → 0` and `packing_floor` activates.  
   WHY: This contradicts the test’s baseline assumption that clamp/floor branches are inactive. The optional Stern-off test would become a saturation/floor MMS, not a clean Stern-off sanity check.  
   WHAT TO DO: Remove `TestSternOffSanity` from this plan, or redesign it as a separate floor/clip-active MMS with different expectations. Do not claim “same L2/H1 rates” under the current no-Stern field.

3. **The Stern coefficient coverage row still overclaims.**  
   WHAT: §6 says `g_S` catches the “value of nondim Stern coefficient.” But §6 also correctly says the nondim conversion is not covered because source and residual both use the converted value.  
   WHY: Those two statements conflict. This MMS catches sign/presence/wiring of the Stern term for the value already present in ctx; it does not validate that the value is physically correct.  
   WHAT TO DO: Change the coverage row to “use of the active ctx Stern coefficient in the residual,” and keep nondim conversion explicitly uncovered.

4. **§5.4 still overstates `A_dyn`.**  
   WHAT: `A_dyn^ex ≪ 10^-5` and “free_dyn is identical to 1 to working precision” is too strong. O₂ alone is about `1.5e-5`, and perturbations make it larger.  
   WHY: The floor-inactive conclusion is fine, but “identical to 1” is not. Worse, it could encourage someone to drop the `(1 - A_dyn)` numerator in the source.  
   WHAT TO DO: Say `A_dyn << 1`, so `free_dyn_floor` is inactive. Keep `1 - A_dyn^ex` exactly in the source.

5. **The `set_phi_applied` reference is probably nonexistent.**  
   WHAT: The plan lists `set_phi_applied` as deferred setter/continuation logic, but I only see live `phi_applied_func.assign(...)` patterns, not a named `set_phi_applied` helper.  
   WHY: Minor, but inaccurate deferred-coverage lists become stale test documentation.  
   WHAT TO DO: Rename this to “runtime mutation of `ctx['phi_applied_func']` / voltage continuation,” unless there is an actual helper you plan to add.

VERDICT: ISSUES_REMAIN