1. **WHAT:** R2 point 8/10 still has the core physics backwards: raising surface `c_H` from pH 10.6 to pH 6 increases acid-form BV rates by enormous factors; it does not “harden” a ceiling.  
   **WHY:** In the live rate law, `(c_H/c_ref)^n` is a multiplier, not a cap. Sulfate acid supply can move the onset/plateau, but it cannot by itself reduce the deep-cathodic current that is already 4× above the deck target at 100 µm.  
   **WHAT TO DO:** Define the ceiling as a transport-limited acid-equivalent flux, not as the BV `c_H^n` factor. The spike must compare BV proton demand against max acid-equivalent supply.

2. **WHAT:** Step 3’s root equation is wrong: `(c_H_new - c_H_old) + (c_HSO4_new - c_HSO4_old) = 0` cannot acidify the surface.  
   **WHY:** Since `c_HSO4` increases with `c_H`, that equation has a root near the old alkaline state or lower `c_H`. It will not produce pH 6-7. If you instead conserve `E = c_H + c_HSO4 - c_OH`, equality to the Phase 6α `E_old` also mostly preserves the old alkaline state.  
   **WHAT TO DO:** Drop the local algebraic pH-shift spike. Use a 1D steady acid-transport model or a flux-cap estimate with bulk H, HSO4, SO4, OH and electrode BV demand.

3. **WHAT:** The revised plan says the relevant buffer is a gradient, but Step 3 still uses a local surface algebra.  
   **WHY:** HSO4 is nearly absent at pH 10.6 and scarce at pH 4. The only plausible contribution is diffusive delivery of bulk HSO4 to the surface, not local equilibrium inventory.  
   **WHAT TO DO:** The spike must include distance `L_eff`, diffusivities, and bulk HSO4 concentration. A per-voltage local root is not a valid branch gate.

4. **WHAT:** Sulfate is a poor pH 6-7 buffer. The plan keeps assuming it can pin that window.  
   **WHY:** pKa2 is ~2. At pH 6, HSO4 fraction is ~1e-4 to 2.5e-5 depending on pKa_eff. Buffer capacity there is tiny.  
   **WHAT TO DO:** Reframe sulfate as an added acid-equivalent transport reservoir from bulk HSO4, not a local pH 6-7 buffer.

5. **WHAT:** Branch A is too permissive: “either pKa endpoint” and `factor >= 1e2` is confirmation-biased and not physically tied to peak formation.  
   **WHY:** The favorable pKa endpoint may be the non-activity-corrected one, and rate amplification is not ceiling binding.  
   **WHAT TO DO:** Branch A should require a robust acid-flux ceiling crossing in the deck voltage window across the pKa bracket, or proceed only as exploratory solver work.

6. **WHAT:** Step 7’s continuation and regression test are mathematically wrong. `Ka_eff = 0` is not the baseline.  
   **WHY:** With `c_HSO4 = c_T*c_H/(c_H + Ka)`, `Ka=0` makes all sulfate HSO4, not all SO4. The static-SO4 baseline is `Ka -> infinity` or zero protonation activation.  
   **WHAT TO DO:** Ramp `q = 1/Ka` from `0` to target, or add an activation `lambda` on protonation. Rename the regression to `TestSulfateActivationZeroReducesToBaseline`.

7. **WHAT:** The “new primary variable `c_T` as a Nernst-Planck species” understates the implementation.  
   **WHY:** `c_T` has no fixed charge or diffusivity. Existing `SpeciesConfig` assumes fixed `z` and `D`; `J_T = J_SO4 + J_HSO4` depends on gradients of both `c_T` and `c_H`.  
   **WHAT TO DO:** Plan a custom residual path, not a normal added species. Include changes to forms, ICs, diagnostics, stoichiometry padding, and observables.

8. **WHAT:** Bulk electroneutrality is not specified after HSO4/SO4 speciation.  
   **WHY:** `[Cs+] = 0.2 M`, `c_T = 0.1 M`, pH 4 is not exactly neutral once a finite HSO4 fraction exists. Small charge errors matter in Poisson/DB initialization.  
   **WHAT TO DO:** Compute bulk Cs from the chosen pKa_eff and pH: `c_Cs = 2c_SO4 + c_HSO4 + c_OH - c_H`.

9. **WHAT:** Step 5 claims deck `j_disk` can imply surface pH near `c_H ~ 1e-3 M`.  
   **WHY:** Disk current alone does not determine surface pH without a transport/kinetic model.  
   **WHAT TO DO:** Use the workbook only for curve shape, onset/peak location, and RRDE selectivity. Do not infer pH from current magnitude.

10. **WHAT:** 6δ.1 is still under-scoped. Alkaline products are OH/HO2, but OH is not an explicit dynamic species and HO2 is absent.  
    **WHY:** Existing reaction-list machinery cannot simply add `R2e_alk`/`R4e_alk` without changing the proton-condition source and likely peroxide chemistry.  
    **WHAT TO DO:** Before 6δ.1, derive source terms for `E = c_H + c_HSO4 - c_OH`, decide whether HO2/H2O2 equilibrium is needed, and verify RHE thermodynamic consistency.

VERDICT: ISSUES_REMAIN