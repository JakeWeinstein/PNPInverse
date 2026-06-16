1. **WHAT:** “Ring magnitude is frame-invariant” is only conditionally true.  
   **WHY:** It is true for a pure post-processing horizontal voltage relabel: `I_ring,pH(V) = I_ref(V - ΔV)`, assuming the full peak remains inside the measured window. It is not automatically true for a solver-side OCP/frame change if `phi_applied`, Stern drop, surface cH, or PNP fields change.  
   **DO:** Define N1 precisely. If N1 is a rigid voltage relabel, test shift-invariant residuals and peak heights over a common window. If N1 changes the actual solver boundary potential, do not call ring magnitude frame-invariant until verified numerically.

2. **WHAT:** Peak-height invariance can fail by finite voltage-window truncation.  
   **WHY:** A rigid shift preserves the global peak height only if the shifted peak is still captured. A pH-dependent slide can make the observed peak look smaller if the peak exits the scanned window.  
   **DO:** Compare ring curves after optimal horizontal alignment on the overlapping voltage support. Use peak height only if the full peak is visibly inside the common window.

3. **WHAT:** The M1/M2 ordering bug is back.  
   **WHY:** M1 says compare A1, but A1 “needs G”; M2 wires G after M1. That is impossible as written.  
   **DO:** Split M1: first N0/N1/A with no G; then M2 wire G; then A1 comparison.

4. **WHAT:** P0.3 says calibrate only pH 6.39 and hold out pH 2 and pH 4, but M3 says fit C to the acid ring collapse.  
   **WHY:** C has no signal at pH 6.39 if it is designed to preserve the locked fit. You cannot estimate C parameters from pH 6.39 alone and still hold out both pH 2 and pH 4.  
   **DO:** Choose an honest protocol: train on pH 4 and predict pH 2, then swap if feasible; or admit C is fit to pH 2/4 and validation is only internal/LOO, not true prediction.

5. **WHAT:** Promoting “C = peroxide consumption” is still too specific.  
   **WHY:** Ring collapse is frame-invariant evidence for pH-dependent peroxide yield/loss, but at one rpm it does not distinguish less H2O2 production, more H2O2 reduction, homogeneous decomposition, altered desorption/escape, or ring response.  
   **DO:** Rename C as “pH-dependent peroxide yield/loss” unless C1/C2 beats equal-complexity competitors such as pH-dependent direct-4e branching or pH-dependent peroxide escape.

6. **WHAT:** A ring-magnitude discriminator is not automatically a disk-branching discriminator.  
   **WHY:** Ring current can change because of disk chemistry, diffusion-layer consumption, homogeneous decomposition, or ring oxidation/collection efficiency. With no N2 H2O2 scans, ring-response artifacts are not independently ruled out.  
   **DO:** Include a ring-efficiency/vertical-scale null or explicitly state that disk-side attribution remains conditional.

7. **WHAT:** C1 still lacks its electrochemical reference details.  
   **WHY:** Specifying `H2O2 + 2H+ + 2e- -> 2H2O` fixes stoichiometry, but the BV term also needs `E_eq`, alpha, concentration factors, and whether reverse oxidation is disabled or thermodynamically represented. These choices affect both current and pH dependence.  
   **DO:** Pre-register C1’s `E_eq`, alpha treatment, reversible/irreversible form, and parameter count before fitting.

8. **WHAT:** The ring-collapse lead gate may lean on the weakest data.  
   **WHY:** The dramatic `0.40 -> 0.079 mA/cm2 below pH 2.3` appears to come from scalar/figure-derived evidence, not the high-fidelity pH 6.39 raw LSV. If that scalar is low-fidelity or convention-mixed, it should not be the sole load-bearing gate.  
   **DO:** Make the digitized pH 2 ring curve the primary C gate. Use the six scalars only as qualitative trend checks or bootstrap-weighted low-confidence features.

9. **WHAT:** “A beats N1” may be untestable from onset alone.  
   **WHY:** If A and N1 are gauge-equivalent horizontal shifts, onset features cannot separate them. A1 may only beat N1 by exploiting surface-cH flexibility, which risks becoming a shape-fitting term rather than evidence for SHE anchoring.  
   **DO:** Treat A vs N1 as a gauge/convention question unless full disk/ring shape residuals provide independent evidence beyond onset position.

10. **WHAT:** The validation language is still too strong for two non-anchor pH curves.  
   **WHY:** With pH 2 and pH 4 only, every choice of which one to train on materially changes the story. LOO is useful, but it will not support strong predictive claims.  
   **DO:** Call M4 “stress testing” or “internal cross-checking,” not validation, unless one non-anchor pH remains untouched through all model selection.

The demotion of A is directionally right. The frame-invariant target should be ring/vertical structure, not onset position. But the plan still needs the N1 definition, train/test protocol, M1/M2 ordering, and C-specific attribution tightened before execution.

VERDICT: ISSUES_REMAIN