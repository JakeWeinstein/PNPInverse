v4 is much better than v3, but it still has blocking architecture errors. The main problem is that it now mixes three incompatible representations: analytic Boltzmann cation, OHP-local conserved pool, and finite-rate volume species.

1. **WHAT:** The plan still double-counts `R_buf`. Lines 290-303 put `+R_buf` into the proton residual and also replace the residual with `E = c_H - c_OH - c_MOH`.  
   **WHY:** In the conserved `E` equation, hydrolysis cancels internally: `+R_buf` in H and `+R_buf` in `c_MOH` give no net source to `E`. Adding it explicitly acidifies twice.  
   **WHAT TO DO:** Either solve individual species equations with `R_buf`, or solve the unified `E` equation with no internal `R_buf` source plus a separate `c_MOH` equation.

2. **WHAT:** `c_M+` is described as “already a Bikerman counterion entry,” but analytic Boltzmann counterions do not have residuals.  
   **WHY:** You cannot add `−R_buf` to a species that is not a solved DOF. “Folds into the existing Bikerman closure” is not an implementation.  
   **WHAT TO DO:** Either modify the analytic closure to solve coupled `M+ / MOH` algebraically at the boundary, or promote cation state to explicit unknowns.

3. **WHAT:** The “OHP-local conserved pool” conflicts with the Boltzmann reservoir.  
   **WHY:** Boltzmann `c_M+` is an instantaneous reservoir determined by potential; an OHP-local fixed total pool is a surface-storage model. Those are different physics.  
   **WHAT TO DO:** Pick one: boundary surface variable with finite storage, or analytic equilibrium closure with no local conservation claim.

4. **WHAT:** The thin-layer source has undefined units. `R_buf * θ(y)` is volumetric, but OHP hydrolysis is interfacial.  
   **WHY:** Total acid produced will scale with the arbitrary chosen `L_OHP`, mesh resolution, and nondimensionalization.  
   **WHAT TO DO:** Use a boundary flux term on `ds`, or normalize the volumetric kernel as `θ/L_OHP` and document units.

5. **WHAT:** `L_OHP = Stern thickness ≈ 5 Å` is physically confused.  
   **WHY:** The Stern layer is represented by a compact-layer boundary condition; it is not part of the PNP solution volume. Adding a 5 Å source inside the domain is a diffuse-side approximation, not Stern chemistry.  
   **WHAT TO DO:** Prefer boundary/OHP algebra. If using a thin layer, call it a numerical reaction layer and verify mesh resolution.

6. **WHAT:** Step 3’s local charge-balance equation is dimensionally wrong. Local charge density `ρ(0)` does not “balance Stern surface charge.”  
   **WHY:** `ρ` is volumetric charge; Stern charge is surface charge. They are related through Poisson integration and boundary conditions, not equality at one point.  
   **WHAT TO DO:** The spike needs a local capacitance/Poisson model, or it should stop claiming self-consistent field feedback.

7. **WHAT:** The spike is still underdetermined. It proposes solving for `(c_M+, c_MOH, c_H, η_local)` from one Phase 6α diagnostic point.  
   **WHY:** You need an equation for how `η_local` changes when cation charge is neutralized. That requires Stern capacitance plus diffuse-layer response, not just algebra.  
   **WHAT TO DO:** Add a reduced Stern/Poisson closure, or skip the spike and go straight to a bounded solver smoke.

8. **WHAT:** The disabled-path regression cannot be “byte-equivalent” if `enable_cation_hydrolysis=True` adds a new `c_MOH` DOF.  
   **WHY:** Extra unknowns change vector layout, residual shape, and solver path even when `λ=0`.  
   **WHAT TO DO:** Require byte-equivalence only when the feature flag is false. For `λ=0`, assert old-variable residual equivalence, not byte equivalence.

9. **WHAT:** `c_MOH` has no well-defined mathematical home. It is “OHP-bound,” appears as `c_MOH(y)`, has no flux, and is forced by a volume kernel.  
   **WHY:** A no-flux volume field can accumulate anywhere the kernel is nonzero; a boundary species needs surface mass terms, not `dx` terms.  
   **WHAT TO DO:** Implement `c_MOH_s` as a boundary/surface unknown, or implement a real thin-layer volume species with storage, IC, and optional diffusion.

10. **WHAT:** The plan still contains stale contradictions: v4 says “This v3,” Branch B says “need Co-Zhang fit,” and open questions still say “Co-Zhang §3 to read.”  
    **WHY:** These stale lines point future implementation back to already-rejected assumptions.  
    **WHAT TO DO:** Clean the artifact before using it as a handoff.

11. **WHAT:** Step 11’s finite-rate water formula is wrong/undefined. `c_H_neutral_water` is not a concentration in the existing model.  
    **WHY:** Water self-ionization finite rate should look like `R = k_f - k_r c_H c_OH` or `k_r(Kw - c_H c_OH)` with clear units.  
    **WHAT TO DO:** Rewrite Phase 6α.1 as a separate derivation before it enters the plan.

12. **WHAT:** Validation risks fitting and validating on the same cation-series data.  
    **WHY:** If `β_M` is tuned per cation against the same CP/IrOx series used for acceptance, passing proves calibration flexibility, not mechanism validity.  
    **WHAT TO DO:** Define calibration targets and holdout targets separately before implementation.

13. **WHAT:** The CP/IrOx source language is still sloppy. Co-Zhang 2019 is not the IrOx-ring method; it used product/ring electrochemistry for CO2RR local pH.  
    **WHY:** Mislabeling the method can send the data audit to the wrong files and wrong calibration equation.  
    **WHAT TO DO:** Attribute IrOx to the Linsey/Ruggiero/Mangan materials only if those files actually contain it.

14. **WHAT:** Starting the spike from Phase 6α `c_H` is now explicitly suspect, but no bracket is added.  
    **WHY:** §6 says fast Kw may over-acidify the Phase 6α state; the hydrolysis spike may inherit that bias.  
    **WHAT TO DO:** Run the spike over a bracket of starting pH values, including no-water-ionization pH and Phase 6α pH, or use measured IrOx local pH when available.

VERDICT: ISSUES_REMAIN