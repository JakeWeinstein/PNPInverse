1. WHAT: Re point 5, the expected F0 amplification ratios are still wrong. With the K2SO4 stack, `C_K_bulk_hat ≈ 166.6`, while v10a has `F0/k_hyd ≈ 52, 67, 101, 129, 226, 414, 732`. So `amplification_from_c_K` is roughly `0.31 → 4.39`, not “both should be ~1.”

WHY: K+ enrichment/depletion is already load-bearing in v10a. Saying “no significant K+ depletion” contradicts the data and will mislead v10b interpretation.

WHAT TO DO: Change expected readout to: `amplification_from_singh ≈ 1`; `amplification_from_c_K < 1` anodic and `> 1` cathodic, reaching several-fold in the negative-V region.

2. WHAT: Re point 4, the R4e log decomposition is still not solver-faithful. In the actual logc/muh forms, with Stern enabled, `eta_raw = phi_applied - phi_boundary - E_eq`, not simply `V_RHE - E_eq`. Also, clip is applied to `eta_scaled` before `α*n_e`; therefore “raw 4e exponent > 100” does not itself mean clipping is active.

WHY: `log_R4e_predicted ≈ log_R4e_measured` can fail for the wrong reason, and the decomposition may blame K0/c_H when the missing term is boundary potential.

WHAT TO DO: Build the diagnostic from the same UFL/log-rate terms as the solver, or label it explicitly as an approximate scalar estimate. At minimum emit boundary `eta_scaled_clipped` avg/min/max and use `phi_applied - phi - E_eq`.

3. WHAT: Case F’s fallback “switch perturbation knob, e.g. perturb φ_applied directly” is not a drop-in replacement for the Stern-capacitance-manifold derivative.

WHY: The selected score is `dRnet/dsigma` along the C_S perturbation manifold. A φ perturbation is a different path and cannot validate the same V_kin rule.

WHAT TO DO: Keep φ perturbation as an auxiliary diagnostic only. For selection, either increase C_S perturbation ε and rerun, or report `no_valid_stern_capacitance_sensitivity` / escalate.

4. WHAT: Case G makes an invalid inference. `no_candidate_passed_locked_rule` plus no artifact flag does not imply “at least one V passes σ<0 and has o2lev<0.9.” The artifact flag only fires when `current_passed=True` and `o2lev>0.9`.

WHY: The next action could be wrong. The real blocker might be current failure everywhere, branch failure everywhere, estimator validity, or σ sign.

WHAT TO DO: In Case G, inspect the per-filter failure matrix directly: counts for `sigma_neg`, `current_passed`, `branch_passed`, `o2lev>0.9`, and estimator validity. Route based on the actual failing filter.

5. WHAT: Routing precedence is inconsistent. Case A says clean V_kin → Phase A.2. The threshold section says route to v10b if every σ<0 fallback-valid V has cap dominance and low sensitivity. A clean but cap-dominated low-sensitivity V could satisfy both.

WHY: The auto-revise step cannot know whether Phase A.2 or v10b wins.

WHAT TO DO: Define precedence explicitly. Recommended: after any returned V_kin, apply transport-artifact check first, then cap-dominance check, then Phase A.2 only if both sanity checks pass.

VERDICT: ISSUES_REMAIN