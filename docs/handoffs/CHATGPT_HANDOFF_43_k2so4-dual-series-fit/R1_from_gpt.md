1. **WHAT:** The OCP shift uses `0.47 + 0.197 + 0.059*pH = 1.0440 V` while the file’s own RHE calibration is `0.549 V`, not theoretical `0.574 V`. The plan says this “does not touch” the OCP convention.  
   **WHY:** It does. `V_solver = V_RHE - V_OCP_RHE` sets the Stern/double-layer state. A 25 mV shift can move surface pH, crowding, and fitted kinetics.  
   **DO:** Put OCP on the same voltage axis as the data. If the 0.47 V component is vs the file’s Ag/AgCl axis, the central shift is likely `0.47 + 0.549 = 1.019 V`, not 1.044 V. Otherwise prove the sign/axis from raw formulas and run a ±25 mV sensitivity.

2. **WHAT:** “Non-converged V dropped” in the objective is not acceptable.  
   **WHY:** The objective becomes parameter-dependent and discontinuous; the optimizer can improve J by failing hard points. The adjoint/FD comparison is then meaningless.  
   **DO:** Treat non-convergence as eval failure or a large penalty. Use a fixed, predeclared experimental mask only. Accepted fits must converge on all objective voltages.

3. **WHAT:** The Stage 1 gate allows `≥11/13` convergence.  
   **WHY:** That is not fit-ready for a dual-series objective. Missing two voltages can erase onset, peak, or plateau information.  
   **DO:** Require full convergence on the fit grid before optimization, or explicitly remove fixed voltages before any fit and keep that mask frozen.

4. **WHAT:** `pc_data` is escaped, collection-corrected H₂O₂ current, but Stage 3 does not prove `form_pc` is the same observable.  
   **WHY:** Local H₂O₂ production is not equal to ring flux if H₂O₂ reduction, sink terms, or boundary loss exist. This directly corrupts the 2e/4e partition.  
   **DO:** Define `pc_model` as net H₂O₂ escape flux matching the RRDE conversion, or prove production equals escape in this model. Add an isolated-reaction ledger test.

5. **WHAT:** `form_cd = electron-weighted sum over ALL active reactions` is underspecified.  
   **WHY:** “All active reactions” can double-count or include artifact/side branches during ablations. Disk current must be exactly the signed electron flux from electrochemical surface reactions, not residual bookkeeping.  
   **DO:** Write the disk-current formula reaction-by-reaction with signs and electron numbers. Add isolated 2e, 4e, and sink tests against ledger current.

6. **WHAT:** The fit window ending at disk zero crossing likely includes non-ORR background.  
   **WHY:** Carbon oxidation, capacitive current, and baseline drift can exist before the zero crossing. The steady ORR model cannot fit that; small σ near onset will bias α/k₀.  
   **DO:** Add blank/background subtraction or a restricted ORR-only window. Define onset/window thresholds from ring and disk separately, not just disk zero crossing.

7. **WHAT:** The σ model is too weak: two cycles, within-bin scatter, one σ floor.  
   **WHY:** Within-bin scatter includes curve slope; two cycles do not estimate systematic uncertainty; disk and ring noise floors are not necessarily the same. Bad weights will drive the fit.  
   **DO:** Estimate σ from cycle-mean differences after binning/interpolation, separate disk/ring floors, add a relative/model-error floor, and report robust/unweighted sensitivity.

8. **WHAT:** AIC is misused.  
   **WHY:** The optimization objective is normalized per series, may drop points, and omits log σ terms. That is not a likelihood. Stage 2 θ* is also a fixed prediction, not a fitted model with the same statistical status.  
   **DO:** Keep J for optimization if desired, but compute raw χ²/log-likelihood on a fixed data vector for statistics. Otherwise call it Δpredictive score, not ΔAIC.

9. **WHAT:** “1-D profile slices” do not establish identifiability unless nuisance parameters are reoptimized.  
   **WHY:** Fixed-parameter slices can fake curvature and miss correlated sloppy directions.  
   **DO:** Use true profile likelihoods: fix one parameter, reoptimize the other three. Also report Hessian/eigenvector covariance for dual vs pc-only.

10. **WHAT:** Unknown `L_eff` is handled only by post-fit sensitivity on accepted θ.  
   **WHY:** `L_eff` changes transport ceilings and the optimum itself. Post-hoc rescoring understates structural uncertainty.  
   **DO:** Refit or profile at each `{12, 15.4, 21.7} µm`, or do not make kinetic-parameter claims until rpm is confirmed.

11. **WHAT:** Processed Sel% and `n_e` are known area-suspect but still appear in QA/gates.  
   **WHY:** The “n_e in [2,4]” check and peak selectivity can be wrong if based on mixed disk/ring-area densities.  
   **DO:** Recompute Sel% and nₑ from raw currents using canonical RRDE formulas. Use processed columns only as provenance cross-checks.

12. **WHAT:** “disk onset V (data ring onset 0.472 V)” conflates two observables.  
   **WHY:** Disk ORR onset and ring peroxide onset need not coincide. A feature gate using the wrong onset target will validate the wrong behavior.  
   **DO:** Define disk onset and ring onset separately with explicit current thresholds.

13. **WHAT:** Ring conversion assumes valid N and mass-limited ring oxidation, but ring hold/calibration is not secured.  
   **WHY:** If the ring was not held at a proper H₂O₂ oxidation plateau, `pc_data` is scale-biased and the 2e/4e split is wrong.  
   **DO:** Get ring hold potential/calibration conditions. Treat N as a refit/sensitivity nuisance, not just a post-fit note.

14. **WHAT:** The 13-point coarse solve is too sparse to fit a 40-bin objective through PCHIP without proof.  
   **WHY:** Peak position and onset gates are ±50 mV; a 13-point grid over ~0.9 V has comparable spacing. The optimizer may fit interpolation artifacts.  
   **DO:** Demonstrate objective/grid convergence at θ*, during final scoring, and near the optimum. Prefer fitting on a denser/adaptive V grid around onset and ring peak.

15. **WHAT:** The “acid-on ablation” is undefined.  
   **WHY:** Re-enabling acid k₀ has no meaning unless the acid parameters and whether water params are held or reoptimized are specified.  
   **DO:** Pre-register the acid parameter source and metric: fixed prior acid values, bounded acid refit, or controlled k₀ sweep.

16. **WHAT:** Stage 2 is overclaimed as “a paper result regardless of outcome.”  
   **WHY:** θ* came from digitized peroxide-only pH 4 data with nonidentified 4e parameters. A bad prediction may reflect θ* uncertainty, not failed transferability.  
   **DO:** Archive it as a pre-registered diagnostic with θ* profile/digitization uncertainty bands. Do not oversell it.

17. **WHAT:** The pH value is treated as concentration without an activity convention.  
   **WHY:** At 0.1 M K₂SO₄, pH is an activity measurement. It probably does not change the “acid negligible” conclusion, but it affects Kw closure and surface-pH interpretation.  
   **DO:** State the activity/concentration convention and, if making surface-pH claims, add a γ_H/Kw_eff sensitivity.

18. **WHAT:** The raw-to-processed cross-check tolerance `≤1e-6 rel` is brittle near zero currents/voltages.  
   **WHY:** Relative tolerance explodes near zero and Excel rounded constants may not reproduce exactly.  
   **DO:** Use combined absolute/relative tolerances, with stricter sign and unit tests.

VERDICT: ISSUES_REMAIN