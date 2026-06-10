1. WHAT: H1 says acid/water handoff is “real system acid-kinetic at onset where local pH≈4,” but Stage B only adds acid-R2e, not acid-R4e.

WHY: If the cathodic flank is still controlled by 4e outcompeting 2e, changing only 2e acid kinetics may fit the bump by distorting peroxide production while leaving the O2 partition mechanism wrong. You cannot attribute failure/success specifically to acid/water handoff.

WHAT TO DO: Test acid-R2e-only and acid-R2e+acid-R4e as separate nested models, or justify chemically/numerically why acid-4e is negligible in the onset window.

2. WHAT: Stage A1 “θ*+acid at locked production k0/α” is not diagnostic enough.

WHY: Locked acid parameters could fail only because the production calibration is incompatible with this dataset/convention, not because H1 is false. Conversely, it could create structure but at an unphysical scale.

WHAT TO DO: Run at least a bounded one- or two-dimensional acid-R2e amplitude sweep before deciding whether the lever moves R1/R2.

3. WHAT: The acid-route rates ∝ c_H² and c_H⁴ are dangerous under the Kw-laundered-H+ closure, but the plan only checks “acid-share per V.”

WHY: Acid share is not enough. The model may create apparent acid kinetics from algebraic water equilibrium rather than transport or real buffer chemistry. That would make a successful H1 fit physically meaningless.

WHAT TO DO: Add a proton-source ledger: net H production/consumption, water-equilibrium implied OH flux, acid-route current, and whether acid current exceeds what true water autoprotolysis/buffer supply could support.

4. WHAT: The H1 signature says acid R2e should add structure in +0.15..+0.35, but onset local pH≈4 and the model pH collapse 9.3→3 happens over +0.27..+0.35.

WHY: The claimed mechanism window is internally inconsistent. If local pH is already alkaline through much of +0.15..+0.27, acid kinetics should be suppressed by c_H² unless the model is laundering H+.

WHAT TO DO: Overlay acid-rate contribution against surface pH before refit. If acid contribution peaks where pH is 8-10, H1 is not chemistry; it is closure artifact.

5. WHAT: H2’s falsification rule “6-param refit leaves R1 with χ² within ~2× of current” is mathematically sloppy.

WHY: “Within 2×” could mean almost anything, and R1 is a positional residual while χ² is amplitude-weighted. A model can improve χ² while still missing the trough/cliff timing.

WHAT TO DO: Define separate acceptance metrics: trough voltage error, cliff voltage error, bump amplitude/sign, left-plateau slope, and global χ².

6. WHAT: The AIC-style criterion is wrong or at least underspecified: “Δχ²·n/2 ≥ 4.”

WHY: If χ² is already a sum over normalized residuals, multiplying by n is double-counting. If “chi2/pt” is mean loss, then yes, multiply by n, but that must be explicit. The factor 1/2 also depends on whether the objective is χ² or negative log likelihood.

WHAT TO DO: State the exact objective. Use ΔAIC = 2Δk - Δχ²_sum for Gaussian residuals, or define an equivalent criterion in terms of the actual minimized scalar.

7. WHAT: The 33 bins are treated as independent Gaussian observations.

WHY: The data came from 754 SVG vertices binned from a plotted curve. Extraction scatter is not experimental uncertainty and adjacent bins are correlated by drawing/rendering/smoothing. AIC and χ² thresholds are not valid as written.

WHAT TO DO: Use the scatter only as a pragmatic weight, not as a statistical likelihood, or estimate correlated/plot-digitization uncertainty and rerun robustness with alternative binning.

8. WHAT: The “bump amplitude ~3σ” claim is overused.

WHY: σ is measurement-scatter from SVG extraction, not replicate experimental noise. A 3σ digitization feature can still be plot artifact, spline artifact, marker overlap, or thresholding.

WHAT TO DO: Require raw xlsx ring data before treating R2 as physically real, or explicitly downgrade the bump to provisional.

9. WHAT: Stage C logistic multiplier uses V directly, not surface potential, overpotential, local field, pH, or adsorbate thermodynamics.

WHY: A fitted V-only multiplier can reproduce any localized feature and will be confounded with the arbitrary −0.903 V OCP shift, reference convention, and Tafel E_ref choices. It is not a Frumkin or quinone model; it is a curve-shape patch.

WHAT TO DO: Parameterize the transition against a physically meaningful variable: electrode potential vs RHE if it is a redox state, Stern/drop/local potential if Frumkin blocking, or local pH if acid/water chemistry.

10. WHAT: Stage C formula `k0_eff = k0·[1 − A·σ((V−Vθ)/w)]` only suppresses 2e at high V for A>0,w>0.

WHY: The observed sequence is local bump, dip, cliff. A monotone suppression cannot generate arbitrary bump-dip behavior unless combined with existing slopes in just the right way. Also the sign may be backwards depending on cathodic/anodic direction.

WHAT TO DO: Precompute the qualitative effect of sign and direction. Include both suppressing and enhancing forms, or define A signed with bounds and require the fitted transition direction to match chemistry.

11. WHAT: Stage C says assign the multiplier to the “k0 R-space Function” per grid point with “NO weak-form change.”

WHY: If k0 enters nonlinear variational forms, solver state, or adjoint tapes, mutating a Function per voltage may not give a valid derivative unless the control graph explicitly includes the multiplier parameters. The statement “outer gradient composes with adjoint dJ/dk0 per point” is not automatically true.

WHAT TO DO: Derive and test the chain rule: ∂J/∂A, ∂J/∂Vθ, ∂J/∂w from stored ∂J/∂k0(V), then finite-difference all three Stage C parameters.

12. WHAT: The logistic width `w` has no bounds or units stated.

WHY: Unbounded `w` can collapse to a step smaller than voltage resolution or inflate into a global slope correction. Both destroy interpretability and identifiability.

WHAT TO DO: Bound `w` in volts, with lower bound above grid/bin resolution and upper bound below the full onset span.

13. WHAT: H3 rejection if `A` saturates 0/1 is invalid.

WHY: A physically complete blocking transition could saturate near 1. Saturation is not evidence against H3; it is evidence the data prefer a near-complete switch or that bounds are too tight.

WHAT TO DO: Reject on failed residual signatures, nonphysical Vθ/w, or instability across bootstrap/binning/profile tests, not on saturation alone.

14. WHAT: H5 only reruns larger L_eff values 21.7/26.2 um.

WHY: If the current cliff is early by 0.04 V, the required correction could need shorter, not longer, effective ionic/O2 transport lengths. One-sided bracketing can miss the sign.

WHAT TO DO: Include shorter and longer film cases, or independently derive the sign of cliff shift with L_eff before spending compute.

15. WHAT: H5 changes one scalar L_eff to mimic species-specific diffusion layers.

WHY: The stated bias is δ_H/δ_O2=1.7 and δ_OH/δ_O2=1.4. A single-film rerun does not test species-specific ionic boundary layers; it changes O2 transport too.

WHAT TO DO: Implement species-specific effective transport or boundary-layer resistance for H/OH separately, even as a reduced diagnostic.

16. WHAT: A3 “bulk H+ 0.1→1.1 mol/m3 = free H+ + full HSO4− pool as protons” is chemically crude.

WHY: Treating the full bisulfate pool as free protons changes ionic strength, electroneutrality, buffer capacity, and sulfate speciation inconsistently. It can create fake pH shifts.

WHAT TO DO: Add an explicit sulfate/bisulfate equilibrium or frame A3 as a deliberately nonphysical upper bound with no mechanistic interpretation.

17. WHAT: The bisulfate ceiling “~0.83 mA/cm2” appears without derivation.

WHY: If that ceiling gates whether proton supply can explain residuals, a wrong stoichiometric/current conversion changes the conclusion.

WHAT TO DO: Show the calculation: concentration, film thickness/flux or inventory, electron/proton stoichiometry, area normalization, and timescale/steady-state assumption.

18. WHAT: R4 total current mismatch is deferred to “parallel data asks,” but total current constrains the 2e/4e competition central to the model.

WHY: You can fit ring peroxide while producing an impossible disk current. Then the inferred 4e outcompetition mechanism is unconstrained and may be wrong.

WHAT TO DO: Make raw disk LSV acquisition a gate before accepting any final chemistry model, or report all fits as peroxide-only non-identifiable.

19. WHAT: The plan accepts a model that improves peroxide shape even if total current remains −5.5 vs ~3.

WHY: That would preserve the main physical inconsistency. A shape fit with wrong electron balance/partition is not a successful ORR model.

WHAT TO DO: Add a hard acceptance criterion on total current once disk data are available, and at minimum a soft sanity bound before Stage C.

20. WHAT: No explicit sign convention checks are included for acid branches, reversible 2e, or logistic modulation.

WHY: The setup has shifted potentials, cathodic negative currents, and formal E_ref values. One sign error in η or the multiplier direction can produce plausible-looking but inverted chemistry.

WHAT TO DO: Add a one-page sign ledger: V_RHE, shifted V, E_ref shift, η definition, cathodic exponential sign, current sign, and expected response to increasing V for each branch.

21. WHAT: The “monotone share ratio ⇒ single interior extremum” claim is asserted but not proven for the actual transport-coupled PDE.

WHY: O2 depletion, pH coupling, Stern response, and nonlinear migration can break simple two-branch algebra. If this claim is wrong, H3 is premature.

WHAT TO DO: Prove it for the reduced model or replace the assertion with numerical evidence from constrained two-branch sweeps/profile fits.

22. WHAT: Stage A4 uses “35 checkpointed fit evals” for stiff/sloppy directions.

WHY: Optimizer trajectory points are not profile likelihoods. They are biased by line search, scaling, and starting point. They do not map identifiability.

WHAT TO DO: Run actual profile slices around θ*: fix each key combination, re-optimize remaining parameters, and plot objective/residual signatures.

23. WHAT: Parameter bounds are missing for Stage B acid k0/α and Stage C A/Vθ/w.

WHY: Without bounds, the optimizer can hide model error in absurd α, unphysical acid rates, near-discontinuous transitions, or negative effective k0.

WHAT TO DO: State all bounds before fitting, with physical rationale and transformations for positivity.

24. WHAT: The plan does not address mesh/grid convergence for new physics beyond “fine grid.”

WHY: The original adjoint gradients were checked for the 4-param water model. Acid pH-sensitive rates and logistic transitions may sharpen layers and change convergence.

WHAT TO DO: Repeat coarse/fine comparison and FD gradient checks for the accepted Stage B/C model, not only for new components.

25. WHAT: Acceptance requires “model becomes non-monotonic in +0.15..+0.35,” but the data need a specific local max-dip-cliff sequence.

WHY: Any wiggle satisfies non-monotonicity, including noise or wrong-sign curvature. That is too weak.

WHAT TO DO: Define feature metrics: local bump voltage, bump height, following dip voltage/depth, cliff onset/slope.

26. WHAT: R3 peroxide re-reduction is dismissed as falsification-only using source-paper topology.

WHY: The observed left plateau mismatch is one of the named residuals. If H2O2 transport/collection or homogeneous decomposition matters, excluding it because one topology omits surface consumption is not enough.

WHAT TO DO: Run a bounded sink diagnostic for H2O2 or prove from independent data that H2O2 consumption is negligible on this carbon under these potentials.

27. WHAT: The plan ignores RRDE collection/transport corrections in the target extraction.

WHY: Ring H2O2 current is not automatically surface H2O2 production. Collection efficiency, peroxide oxidation kinetics at the ring, delay, and thresholded zero tail can distort the apparent curve.

WHAT TO DO: State the conversion from modeled H2O2 flux to target current and verify collection efficiency/units/sign against the original RRDE setup.

28. WHAT: The thresholded-zero tail is included in the target but not handled specially in the objective.

WHY: Treating thresholded zeros as exact Gaussian data overweights censored observations and can pull the onset/cliff.

WHAT TO DO: Use censored-data loss or downweight/exclude flagged tail points in sensitivity runs.

29. WHAT: The OCP convention risk is underplayed: “confirm the 0.47 V OCP component” is parallel, not gating.

WHY: If the −0.903 V shift or OCP component is wrong, every E_ref/onset interpretation and H3 Vθ window is shifted.

WHAT TO DO: Make OCP/reference verification a pre-Stage-B gate.

30. WHAT: No cross-validation or bootstrap is planned.

WHY: With 6-9 parameters and correlated extracted data, improvement can be overfit to digitization artifacts, especially the bump.

WHAT TO DO: Bootstrap bins/vertices, vary bin count, omit the bump region, and check whether parameters and residual signatures persist.

31. WHAT: “Existing Phase-6 ablation matrix then runs at θ_final” is too late.

WHY: If accepted θ_final only works because of one fragile closure or artifact, late ablation becomes documentation, not decision support.

WHAT TO DO: Run minimal ablations immediately after each candidate passes Stage B/C acceptance, before declaring it final.

32. WHAT: The plan lacks an explicit null model.

WHY: A generic voltage-localized basis or spline multiplier may fit the bump as well as the chemistry stories. Without a null, H3 can be overinterpreted.

WHAT TO DO: Compare H3 against a deliberately phenomenological smooth voltage correction with the same parameter count, and only claim chemistry if H3 is uniquely constrained.

VERDICT: ISSUES_REMAIN