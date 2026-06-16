1. **WHAT:** R2-#7 fix is impossible as written. Calling `collect_v10a_rung_diagnostics(ctx)` after `solve_lambda_ramp_from_warm_start` returns only sees the final ctx, usually λ=1. It cannot reconstruct `warm_reconverge` or λ=0 states.  
   **WHY:** Your λ=0/warm diagnostics would be mislabeled final-state diagnostics. That is worse than missing data.  
   **WHAT TO DO:** Patch the solver helper to run diagnostics/callback at `warm_reconverge` and λ=0, or explicitly exclude those rungs from the diagnostic surface.

2. **WHAT:** R2-#8 partial-rung capture still misses failures in `warm_reconverge` and λ=0. The callback only fires in the positive-λ loop.  
   **WHY:** Early failures still produce only an exception string, exactly the failure mode you claimed to fix.  
   **WHAT TO DO:** Either modify `solve_lambda_ramp_from_warm_start` to callback on warm/λ0 solves, or document that partial capture only covers positive-λ failures.

3. **WHAT:** R2-#6 says `cd_mA_cm2`, `pc_mA_cm2`, `x_2e`, and `o2_flux_levich_ratio` are “already in rung_diag.” They are not. The helper emits `cd_observable` nondim and per-reaction currents; v10a' driver derives the rest outside the rung diagnostics.  
   **WHY:** The routing/plot fields will be missing unless A.2 recomputes them.  
   **WHAT TO DO:** Add explicit A.2 computations for `cd_mA_cm2`, `pc_mA_cm2`, `x_2e`, `o2_flux_levich_ratio`, and `current_filter_ratio`.

4. **WHAT:** R2-#1 `picard_converged = iters < 8 AND last_rel < 1e-4` is not robust. A Picard loop can converge on iteration 8, and `gamma_picard_history` does not store the pre-update γ needed to compute `last_rel` when `iters == 1`.  
   **WHY:** You will false-fail valid rungs and may still misclassify one-iteration cases.  
   **WHAT TO DO:** Emit `gamma_picard_rel_history` or `gamma_picard_converged` from the solver helper. If avoiding solver edits, compute rel only when history length ≥2 and treat length-1 cases separately.

5. **WHAT:** The new transition pass criterion is internally inconsistent. The adopted grid predicts θ≈0.157 at `3e-5` and θ≈0.861 at `1e-3`, so the set `{3e-5,1e-4,2e-4,5e-4,1e-3}` does not span `[0.05,0.9]`.  
   **WHY:** The criterion may be impossible even when the run behaves exactly as predicted.  
   **WHAT TO DO:** Add `1e-5` and include `2e-3` in the transition-span check, or redefine the criterion as “at least three points inside 0.1<θ<0.9.”

6. **WHAT:** `required_kdes_Gamma_max` from a k_hyd ramp is not identifiable. Once saturated, changing `k_hyd` barely changes `R_net`; it gives no derivative with respect to `k_des·Γ_max`.  
   **WHY:** The proposed v10b scaling prior is numerically unsupported. Selectivity is nonlinear and transport-coupled; “deck target / A.2 saturated” is not a valid source-scaling factor.  
   **WHAT TO DO:** Either run explicit small `k_des` or `Γ_max` perturbations near saturation, or report only the observed selectivity gap with no inferred scaling.

7. **WHAT:** “If selectivity is within 10 pp, v10b only needs C_S” is not justified. `Γ_max` and `k_des` are smoke placeholders awaiting literature calibration.  
   **WHY:** A one-voltage smoke match can be accidental, especially before plumbing ablations. Skipping Γ/k_des calibration undermines the locked sequence.  
   **WHAT TO DO:** Keep v10b mandatory for literature calibration; A.2 may prioritize what v10b should examine, not cancel it.

8. **WHAT:** The transport gate is applied at the highest converged k_hyd. That is the most likely point to be transport-contaminated and may not be the best saturated-but-kinetic point.  
   **WHY:** You could declare A.2 inconclusive even if `1e-2` is already saturated enough and still transport-clean.  
   **WHAT TO DO:** Route using the highest k_hyd satisfying both saturation and `o2_flux_levich_ratio < 0.9`; separately report transport re-entry above it.

9. **WHAT:** R2-#16 still overclaims. `amp_from_singh(k_hyd)<2` at fixed `r_H_El=200.98` does not prove v10b can keep the K⁺ Cu prior.  
   **WHY:** Fixed-value amplification is not parameter sensitivity. If `r_H_El` is wrong, A.2 will not see it.  
   **WHAT TO DO:** Say only “Singh amplification is small under the current prior.” Do not make a recalibration/no-recalibration decision without an `r_H_El` perturbation or literature argument.

10. **WHAT:** The mass-balance policy contradicts itself. One section says residual violation “don’t fail-stop,” later escalation says “debug; don’t proceed to routing.”  
    **WHY:** Implementers need one behavior.  
    **WHAT TO DO:** Make it a hard gate for λ=1 baseline and routing rungs; otherwise label non-routing rungs as diagnostic failures.

11. **WHAT:** The plan still says “λ ladder: 5 rungs,” but `AdaptiveLadder` can insert additional λ values.  
    **WHY:** Output schema, plotting, and pass logic can silently assume the wrong rung set.  
    **WHAT TO DO:** State “initial ladder is 5 rungs; actual ladder may include inserted rungs,” and make JSON/plots consume arbitrary λ values.

12. **WHAT:** “Picard should converge in 2-4 iters at high k_hyd” is asserted without evidence.  
    **WHY:** High-k stiffness is exactly the unknown being characterized. This sentence will bias debugging toward λ/Newton even if Picard coupling is the problem.  
    **WHAT TO DO:** Remove the prediction or mark it as a hypothesis tested by `gamma_picard_history`.

VERDICT: ISSUES_REMAIN