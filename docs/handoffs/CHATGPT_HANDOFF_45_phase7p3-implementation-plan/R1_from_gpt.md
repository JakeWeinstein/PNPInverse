1. **WHAT:** The phrase “pH axis is a second identifying dimension” is too strong.  
   **WHY:** pH variation does not break the one-rpm direct-4e vs series-2e+2e degeneracy; it only tests whether a pH-coupled family can explain trends. A wrong pH-dependent direct-4e, peroxide escape, collection-efficiency, or activity model can still pass.  
   **DO:** Reframe as “pH-trend sufficiency, not identification.” Add same-parameter-count nulls/competitors.

2. **WHAT:** G1 is not decisive as written. A0 almost hard-wires a positive RHE slope through `E_eq = E0_SHE + 0.0592 pH`.  
   **WHY:** Passing G1 may prove only that the frame transform was imposed correctly, not that the SHE-anchored mechanism is true. Failing G1 may instead expose OCP/frame/digitization error.  
   **DO:** Require A0/A1 to beat an RHE-flat null and an OCP-shift null under the same onset extractor.

3. **WHAT:** The sequencing is internally inconsistent: A1 “needs G,” but G is Phase 2 after the A gate.  
   **WHY:** If A0 overpredicts +59 mV/pH, you cannot complete G1 without surface-cH wiring.  
   **DO:** Sequence as: P0.1 → A0 → implement G → A1 → G1.

4. **WHAT:** P0.1 is not optional risk mitigation; it is the central falsification guard.  
   **WHY:** A pH-dependent potential-frame mistake can fake exactly the onset slope you are trying to explain.  
   **DO:** Make byte-equivalence at pH 6.39 a hard gate for every reaction, not just the 2e route. Use `E0_SHE,j = Eeq_locked,j - 0.0592*6.39`.

5. **WHAT:** “Refit `{E0_SHE, k0}` at calibration pH” conflicts with preserving the locked pH-6.39 fit.  
   **WHY:** If k0 moves, preservation becomes soft regularization, not a constraint.  
   **DO:** For A0/G1, freeze the locked parameters and only transform the reference frame. Later fits may add parameters, but pH 6.39 residual must stay within a fixed tolerance.

6. **WHAT:** The fit/test split leaks information. Holding out pH 6 while anchoring pH 6.39 is basically not a real holdout.  
   **WHY:** It will overstate predictive success near the anchor.  
   **DO:** Treat pH 6 as an anchor-neighborhood check, not validation. The meaningful held-out tests are pH 2 and pH 4, but do not use either for model selection if you later claim prediction.

7. **WHAT:** The anti-overfitting argument “parameter-count ≪ points” is weak.  
   **WHY:** Digitized curve points are highly autocorrelated; effective N is curve-level/feature-level, not 700 independent observations.  
   **DO:** Score by curve-level features plus residual bands, bootstrap digitization uncertainty, and compare against equal-complexity alternatives.

8. **WHAT:** C1 is under-specified stoichiometrically.  
   **WHY:** “Consumes peroxide, adds disk current” is not enough. H2O2 electroreduction needs electron count, H+ stoichiometry, boundary flux signs, current accounting, and possibly H+ consumption.  
   **DO:** Specify the reaction explicitly, e.g. `H2O2 + 2H+ + 2e -> 2H2O`, then unit-test Faradaic current vs H2O2/H+ flux.

9. **WHAT:** C2 can become a curve-fitting peroxide sink.  
   **WHY:** A homogeneous H2O2 sink plus O2 regeneration can suppress ring current and increase disk current in a way that mimics direct 4e ORR.  
   **DO:** Constrain C2 by physical rate scale/Damkohler length, test C2-only, and require mass balance plus realistic spatial source profiles.

10. **WHAT:** G2 is not decisive.  
   **WHY:** If C fails, it may be because H/bisulfate/activity/collection effects are missing, not because peroxide consumption is false. If C passes, a wrong flexible sink may have snuck through.  
   **DO:** Make G2 a sufficiency gate only. Require simultaneous disk/ring shape, peak position, H2O2 balance, pH-6.39 preservation, and physically bounded rates.

11. **WHAT:** H/bisulfate is placed too late.  
   **WHY:** The acid-end collapse is exactly where bisulfate/activity uncertainty matters most. Treating H only as a late sensitivity can falsely reject or over-credit C.  
   **DO:** Include an acid-end H bracket during G2, or mark pH 2 as qualitative until bracketed.

12. **WHAT:** A1 `m` and C1 acid order are likely confounded.  
   **WHY:** Both use surface cH powers and can trade off onset/branching changes, especially when fitted jointly.  
   **DO:** Fit A only on low-current disk onset first, freeze it, then fit C to ring/selectivity. Joint refit only as a final sensitivity.

13. **WHAT:** “Surface cH” needs an exact reaction-plane definition.  
   **WHY:** Boundary trace, OHP, Stern-plane, and bulk-adjacent cH are not interchangeable. The inferred `m` is meaningless if the code samples the wrong variable.  
   **DO:** Document the exact UFL expression used by BV rates and add a test comparing imposed bulk/surface cH gradients.

14. **WHAT:** Selectivity conventions are still a trap.  
   **WHY:** The scalar selectivity data is area-mixed and ring-overweighted; using it in G2 can reward the wrong model.  
   **DO:** Fit raw disk/ring where possible. Treat selectivity/n_e as diagnostics under explicitly matched conventions.

15. **WHAT:** The thesis still overclaims “one shared first ET” while the implementation keeps parallel 2e/4e routes.  
   **WHY:** The code approximation cannot prove the shared-intermediate mechanism; it only tests a shared pH-shifted onset spine.  
   **DO:** Rename the claim: “phenomenological SHE-anchored onset spine plus peroxide-consumption branch.” Save “shared first ET” for a later actual intermediate model.

16. **WHAT:** The first executable milestone should be narrower.  
   **WHY:** “P0.1+P0.2 + A0 driver + G1” still mixes frame validation, onset extraction, model fitting, and hypothesis testing.  
   **DO:** Hard order: frame byte-test first; onset extractor second; A0 null comparison third; only then decide whether G/A1 is worth building.

VERDICT: ISSUES_REMAIN