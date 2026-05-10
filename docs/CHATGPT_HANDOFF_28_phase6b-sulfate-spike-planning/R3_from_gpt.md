1. **WHAT:** Step 7’s source term is not an equilibrium closure; it is an always-positive acid generator. `R_M = c_M_total/(1 + c_H/Ka)` is the equilibrium concentration of neutral `M(OH)`, not a net reaction rate.  
   **WHY:** Adding `+k_eq R_M` to the H residual creates protons without the reverse reaction or cation/neutral bookkeeping. Newton will drive pH artificially acidic.  
   **WHAT TO DO:** Use either a true finite-rate residual `k_f c_M+ - k_r c_MOH c_H`, or an algebraic shadow species with a conserved proton condition.

2. **WHAT:** `c_H ≈ Ka_M` contradicts the “cation reservoir not depleted” assumption.  
   **WHY:** At `pH = pKa`, `[M(OH)]/[M+] = 1`, so half the OHP cation is neutral. That is major depletion of charged cation, not negligible.  
   **WHAT TO DO:** Track the charged fraction: `c_M+ = c_M,total * c_H/(c_H + Ka)` and `c_MOH = c_M,total * Ka/(c_H + Ka)`.

3. **WHAT:** Poisson charge is wrong if hydrolysis is active.  
   **WHY:** Neutral `M(OH)` does not contribute +1 charge. If Cs is 50% hydrolyzed at the OHP, the local positive charge density drops by ~50%, which changes the field that caused the pKa shift.  
   **WHAT TO DO:** Feed only `c_M+` into Poisson; feed both `c_M+` and `c_MOH` into packing if neutral CsOH remains in the layer.

4. **WHAT:** The proton-condition residual should be unified, not “Phase 6α plus additive H source.”  
   **WHY:** Cation hydrolysis is an acid-base rearrangement. For `M+ + H2O ⇌ M(OH)^0 + H+`, the conserved coordinate is `E = c_H - c_OH - Σ c_MOH`, not `c_H - c_OH` plus a source.  
   **WHAT TO DO:** Implement `E = c_H - c_OH - c_MOH` and derive `J_E`; if `M(OH)` is OHP-bound, make it a boundary/surface term, not a volume source.

5. **WHAT:** The plan applies OHP chemistry “at point y” across the domain.  
   **WHY:** The pKa collapse is an interfacial/OHP effect, not a bulk volume reaction. Applying it wherever the analytic cation exists will acidify the diffuse layer and bulk incorrectly.  
   **WHAT TO DO:** Localize hydrolysis to the reaction plane/OHP as a boundary algebraic condition or a thin-layer model with explicit thickness.

6. **WHAT:** Step 10 points to the wrong source for `Ka_M(φ)`.  
   **WHY:** Zhang & Co 2019 validates cation-dependent local pH trends in CO2RR, but its “Section 3” is a local-pH measurement derivation, not the hydrolysis pKa functional form. It cites Singh et al. for the pKa-shift hypothesis.  
   **WHAT TO DO:** Read Singh/Kwon/Lum/Ager/Bell 2016 and its SI for the field/pKa model; use Zhang & Co only as experimental support for cation ordering.

7. **WHAT:** The pKa table is not a universal solver input.  
   **WHY:** The cited Cs pKa drop is for a specific cathode/potential environment, e.g. Ag at strong cathodic bias in CO2RR literature. ORR on carbon with Stern layer and RHE potentials may see a different shift.  
   **WHAT TO DO:** Treat near-cathode pKa as a calibrated function of local OHP field/potential, not a constant per cation.

8. **WHAT:** The algebra spike is almost tautological.  
   **WHY:** If it simply assigns pH ≈ literature near-cathode pKa, it will “reproduce” the cation series by construction and will not test the model.  
   **WHAT TO DO:** Make the spike compute actual `c_M+`, `c_MOH`, charge loss, and predicted pH from the Phase 6α OHP potential/cation density.

9. **WHAT:** `δ_pKa = 0` is not a byte-equivalent disabled path.  
   **WHY:** Bulk pKa still permits small but finite hydrolysis, and the current proposed source term would still be nonzero.  
   **WHAT TO DO:** Add an explicit activation `lambda_hydrolysis`; the regression should use `lambda_hydrolysis = 0`, not `δ_pKa = 0`.

10. **WHAT:** The OHP cation-density claim needs correction before it becomes a design constant.  
    **WHY:** The code’s Cs steric parameter implies a hard-sphere cap closer to ~37 M physical, not necessarily 25 M, and actual surface density may be much lower than the cap. Existing `iv_curve.json` does not store `c_counterion0_surface_mean`.  
    **WHAT TO DO:** Rerun one point with diagnostics or persist counterion diagnostics; use actual `c_Cs(OHP)`, not cap density.

11. **WHAT:** The plan assumes CsOH neutral fate is ignorable, but that fails exactly in the regime you want.  
    **WHY:** If pH is near pKa, neutral fraction is O(1). Neutral species diffusion/desorption can deplete the reservoir and alter packing.  
    **WHAT TO DO:** For 6β.1 either constrain `M_total` as an OHP-local conserved pool, or add a finite exchange timescale with bulk `M+`.

12. **WHAT:** “With Cs pH 4-5, cathodic decay may emerge naturally from acid-form ORR + transport” is still unsupported.  
    **WHY:** Acid-form ORR with higher `c_H` amplifies cathodic rate; transport can plateau, not generally decay. The redirect fixes pH source, not the decay mechanism.  
    **WHAT TO DO:** Keep 6δ as likely, not merely deferred. Require the 6β.1 sweep to demonstrate decay before dropping alkaline/coverage kinetics.

13. **WHAT:** The CP comparison direction is mostly right, but the data must be handled carefully.  
    **WHY:** CP fixes current and measures potential, so higher activity means less cathodic V_RHE at fixed |j|. But local CSV values are vs Ag/AgCl, include outliers, and do not show a clean Cs>K>Na>Li ordering at every current.  
    **WHAT TO DO:** Convert to RHE, QC outliers, use replicate/error data from the `.mat` files, and compare trends by pH/current regime, not one monotone ordering.

14. **WHAT:** Validation needs more parameters than Stokes radius and bulk concentration.  
    **WHY:** Cation-specific OHP behavior depends on effective closest approach, Stern/OHP location, `a_nondim`, `phi_clamp`, pKa_bulk, pKa_shift law, dielectric/local field model, and possibly activity coefficients.  
    **WHAT TO DO:** Add these to the config schema per cation; do not hide them inside hard-coded counterion dicts.

15. **WHAT:** The plan overstates literature certainty.  
    **WHY:** Ruggiero 2022 reports cation-dependent local pH buffering in ORR and discusses cation hydrolysis as a hypothesis; Zhang & Co supports cation-dependent local pH in CO2RR; Singh 2016 provides the hydrolysis model. None directly validates the exact algebraic OHP closure proposed here.  
    **WHAT TO DO:** Phrase 6β.1 as a mechanistic implementation to test, not as settled chemistry.

Sources checked: Ruggiero et al. 2022 ORR manuscript via OSTI, especially local-pH/cation sections; Zhang & Co 2019 Angewandte DOI `10.1002/anie.201912637`; Singh et al. 2016 JACS DOI `10.1021/jacs.6b07612`.

VERDICT: ISSUES_REMAIN