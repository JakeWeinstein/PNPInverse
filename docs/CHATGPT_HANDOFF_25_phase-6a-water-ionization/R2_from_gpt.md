1. **Point #6: your NP flux is missing diffusivity in electromigration.**  
   **WHAT:** You wrote `J_H = -D_H ∇c_H - c_H ∇φ` and `J_OH = -D_OH ∇c_OH + c_OH ∇φ`. Standard NP is `J_i = -D_i(∇c_i + z_i c_i ∇φ)`.  
   **WHY:** Your `J_E` migration coefficient is wrong. It should be `D_H c_H + D_OH c_OH`, not `c_H + c_OH`.  
   **DO:** Use:
   ```text
   c_OH = Kw / c_H
   J_E = J_H - J_OH
       = -(D_H + D_OH Kw/c_H^2) ∇c_H
         -(D_H c_H + D_OH Kw/c_H) ∇φ
   ```
   In log form:
   ```text
   J_E = -(D_H c_H + D_OH c_OH)(∇u_H + ∇φ)
   ```
   In muh form:
   ```text
   J_E = -(D_H c_H + D_OH c_OH) ∇μ_H
   ```

2. **Point #6: your weak form must stay conservative.**  
   **WHAT:** You propose `[exp(2u)+Kw] u_t + exp(u) ∇·J_E = 0`. That is the strong equation multiplied by `c_H`.  
   **WHY:** If implemented naively and integrated by parts, this changes the weak form and breaks conservation.  
   **DO:** Implement the conservative residual:
   ```text
   ∂E/∂t + ∇·J_E = 0
   E = c_H - c_OH
   ```
   Weak form:
   ```text
   ∫ v (c_H + c_OH) u_t dx - ∫ ∇v · J_E dx + ∫ v J_E·n ds = 0
   ```

3. **Point #6/#12: boundary conditions are still missing.**  
   **WHAT:** The proton-condition PDE needs explicit BCs. You only described the interior flux.  
   **WHY:** The surface BV coupling is where the model lives. Wrong sign or wrong stoich will corrupt current and pH.  
   **DO:** State BCs explicitly:
   - Electrode: `J_E·n = J_H,BV·n - J_OH,Faradaic·n`; currently `J_OH,Faradaic = 0`.
   - For both acidic 2e and 4e ORR, H+ consumption is 1 H+ per electron.
   - Bulk/top: Dirichlet `c_H = c_H_bulk`, hence `E = c_H_bulk - Kw/c_H_bulk`.
   - Sides: no-flux `J_E·n = 0`.

4. **Point #12: muh formulation needs its own derivation, not a port.**  
   **WHAT:** In muh, `c_H = exp(μ_H - φ)`, so storage depends on both `μ_H` and `φ` if any transient/pseudo-time term exists.  
   **WHY:** Reusing the logc residual with symbol substitutions can introduce an unphysical missing `φ_t` term.  
   **DO:** Derive muh separately. For steady solves, flux is clean: `J_E = -(D_H c_H + D_OH c_OH)∇μ_H`. For transient forms, write the storage term explicitly.

5. **Point #1/#2: Damköhler arithmetic is improved but still sloppy.**  
   **WHAT:** Bulk relaxation is `1/(1.4e7 s^-1) ≈ 70 ns`, not `0.7 μs`. At pH 14, `c_H = 1e-14 M`, `c_OH = 1 M`; not “both ~1 M”. Relaxation there is ~7 ps, not 7 ns.  
   **WHY:** The conclusion survives, but the plan still contains bad arithmetic in the evidence section.  
   **DO:** Fix the numbers and separate “minor arithmetic” from the actual dimensionless argument.

6. **New blocker: finite water source capacity is still not checked.**  
   **WHAT:** Fast equilibrium does not mean infinite proton production. The maximum positive net source in the finite-rate model is `k_r Kw ≈ 1.4e-3 M/s`.  
   **WHY:** A deck-scale current of `0.18 mA/cm²` over `100 μm` requires roughly `1.9e-4 M/s`, already ~13% of that maximum if supplied volumetrically. At `16 μm` or larger currents it can become comparable to or exceed the finite-rate source. Option C may over-supply protons.  
   **DO:** Add a validation scalar:
   ```text
   R_required / (k_r Kw)
   ```
   over the sweep. If it is not comfortably small, Option D is not optional.

7. **Point #15: `c_OH_clamp` breaks the model.**  
   **WHAT:** A hard clamp on `c_OH` violates `c_H c_OH = Kw`, breaks the conservative E equation, and makes the local-equilibrium test fail or become meaningless.  
   **WHY:** You cannot claim fast-water equilibrium and then silently cap the equilibrium species.  
   **DO:** Use continuation as the primary stabilization. If a clamp ever activates, mark the run invalid. If needed, use a smooth numerical regularization and report it explicitly.

8. **Point #15: the `exp(-2u)` stiffness is partly self-inflicted.**  
   **WHAT:** In `∇c_H` form you see `D_OH Kw exp(-2u)`. In log or muh flux form the coefficient is `D_H c_H + D_OH c_OH`, i.e. `D_H exp(u) + D_OH Kw exp(-u)`.  
   **WHY:** Implementing the c-gradient form makes Newton look worse than necessary.  
   **DO:** Implement the log/muh conservative flux directly.

9. **Point #17: local equilibrium check is tautological.**  
   **WHAT:** If `c_OH` is reconstructed as `Kw/c_H`, then `c_H c_OH / Kw = 1` by construction.  
   **WHY:** This test will pass even if the E residual, flux signs, or boundary flux are wrong.  
   **DO:** Keep it as a plumbing sanity check only. Real validation needs E conservation, current balance, Yash profile comparison, and finite-source-capacity estimates.

10. **Point #17: mass-conservation acceptance is under-specified.**  
    **WHAT:** “Integrate ∂E/∂t over the domain; should equal net surface BV current” lacks sign, units, nondim scaling, and boundary terms.  
    **WHY:** This is easy to pass incorrectly.  
    **DO:** Write the exact nondim identity with the same normal convention used in Firedrake and include top/sides/electrode boundary contributions.

11. **Point #19 defense on sulfate is weak.**  
    **WHAT:** Saying sulfate buffering matters only near pH 1-3 is not enough. At pH 4, ~1 mM HSO4- exists against only 0.1 mM free H+. That reservoir is not automatically negligible.  
    **WHY:** It can affect the same proton-supply bottleneck you are trying to fix.  
    **DO:** Defer sulfate only after comparing its buffer capacity/current capacity against water autoionization over the target current range.

12. **Thermodynamic consistency with Bikerman is unresolved.**  
    **WHAT:** You enforce `c_H c_OH = Kw` while also putting OH- into a steric/activity-like Bikerman closure.  
    **WHY:** If the model treats steric effects as activity corrections, water equilibrium should use activities, not bare concentrations.  
    **DO:** Decide explicitly: either this is a concentration-Kw model with sterics only in Poisson/packing, or derive `a_H a_OH = Kw` using the same γ convention.

13. **Point #11 IC plan is still too hand-wavy.**  
    **WHAT:** “Use Picard-state H+ profile directly” is not an implementation spec.  
    **WHY:** The IC can easily violate the E residual or seed incompatible φ/c profiles.  
    **DO:** Specify the exact IC reconstruction order: compute `c_H(y)`, set `c_OH=Kw/c_H`, rebuild charge density with OH-, recompute/verify φ consistency, then verify residual norms.

14. **Point #20 expression-shape regression is fragile.**  
    **WHAT:** “No `Kw_hat` constant in UFL expression” is not a robust test; constants can be folded, renamed, or hidden.  
    **WHY:** It tests implementation details, not behavior.  
    **DO:** Prefer disabled-path numerical equivalence plus a config-level assertion that water-ionization branches are not registered.

15. **Yash cross-check needs comparability gates.**  
    **WHAT:** Yash has 6 dynamic species and possibly different electrolyte/BC handling.  
    **WHY:** A failed profile comparison may reflect model differences rather than an Option C bug.  
    **DO:** Before using it as acceptance evidence, lock voltage, pH, species, coordinate scaling, bulk concentrations, and boundary conditions.

VERDICT: ISSUES_REMAIN