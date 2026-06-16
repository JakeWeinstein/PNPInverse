1. **WHAT:** “No Picard iteration needed within a rung” is false. `solve_lambda_ramp_from_warm_start` still runs an outer Γ Picard loop, and only `snes_converged` is treated as success.  
   **WHY:** High `k_hyd` can produce SNES success with Γ Picard not actually converged. Your pass criterion can accept stale Γ.  
   **WHAT TO DO:** Record `gamma_picard_iters`, detect max-iter exits, and add a Γ residual check: `R_forward_capped - denominator_kprot*gamma - k_des*gamma ≈ 0`.

2. **WHAT:** The A.2 grid does not actually locate cap onset. At V_kin, baseline `F0/k_hyd ≈ 291`; cap half-onset is near `k_hyd ≈ Γ_max/291 ≈ 1.6e-4`, and θ>0.9 near `1.5e-3`. Your grid jumps `1e-4 → 1e-3 → 1e-2`.  
   **WHY:** You miss the transition shape and waste 3 rungs on a saturated plateau.  
   **WHAT TO DO:** Use `{3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1}` or compute an adaptive grid from the measured λ=0/λ=1 `forward_avg_no_k_hyd`.

3. **WHAT:** The expected table is numerically wrong at `k_hyd=1e-4`. Using the v10a' `F0_avg=0.291` at `1e-3`, the predicted `F0≈0.0291`, `denom_cap≈0.619`, `θ≈0.38`, `Γ≈0.018`, not `Γ≈5e-3`, `θ≈0.10`.  
   **WHY:** This corrupts the claimed “sub-smoke linear regime” check. `1e-4` is not safely linear.  
   **WHAT TO DO:** Correct the table and add at least `1e-5` or `3e-5` if you want a real linear-regime point.

4. **WHAT:** `denom_cap/total > 0.8 AND θ > 0.9` is redundant at λ=1. Algebraically, `denom_cap/total == θ` for the emitted v10a denominator.  
   **WHY:** The routing rule double-counts one signal and pretends it has two independent cap-dominance criteria.  
   **WHAT TO DO:** Replace with one coverage/cap metric plus an independent metric, e.g. actual Stern sensitivity, transport ratio, or `d log R_net / d log k_hyd`.

5. **WHAT:** The “cap-dominated routing” still depends on `|sensS| < 0.10`, but A.2 skips the perturbation column by default.  
   **WHY:** The rule is not executable in the default run.  
   **WHAT TO DO:** Either compute perturbations for the few λ=1 transition/plateau rungs, or remove `|sensS|` from A.2 routing and explicitly defer that decision.

6. **WHAT:** The v10b bridge is under-specified. A.2 mostly proves the smoke plateau `R_net ≈ k_des·Γ_max = 0.047`, which is already known from constants.  
   **WHY:** v10b needs calibration-relevant outputs: how much source is needed to move selectivity/current/pH, not just that the placeholder cap saturates.  
   **WHAT TO DO:** Add per-`k_hyd` `x_2e`, `cd`, `pc`, `o2_flux_levich_ratio`, `c_H_avg`, `c_K_avg`, and a derived “required `k_des·Γ_max` scale” or explicitly downgrade A.2 to numerical cap characterization only.

7. **WHAT:** The plan says “collect every rung’s diagnostics,” but the helper does not attach `collect_v10a_rung_diagnostics` to `warm_reconverge` or the λ=0 rung.  
   **WHY:** Your promised `{k_hyd × λ}` JSON will have missing λ=0 diagnostics unless the driver adds them.  
   **WHAT TO DO:** Manually augment λ=0 records after the floor solve, or state λ=0 is convergence-only and excluded from the diagnostic surface.

8. **WHAT:** `LadderExhausted` loses structured partial results unless the driver captures them through `rung_callback`.  
   **WHY:** The plan promises diagnostics and λ history for failures, but the current wrapper pattern returns only an error string.  
   **WHAT TO DO:** Use a side-channel rung collector callback and persist partial rungs on exception.

9. **WHAT:** The pass criterion “≥5/6 converge” conflicts with the failure tree. It could pass while `k_hyd=1e-3` or `1e-2` fails.  
   **WHY:** Baseline reproduction and transition convergence are mandatory; a generic count is not enough.  
   **WHAT TO DO:** Require convergence for `1e-3`, at least two transition points around onset, and at least one saturated high-`k_hyd` point.

10. **WHAT:** The warm-walk path to V_kin is not specified tightly enough.  
    **WHY:** A direct `+0.55 → -0.10` jump may not reproduce v10a' and may fail differently.  
    **WHAT TO DO:** Reuse the exact v10a' voltage path through V_kin, at minimum `{+0.55,+0.40,+0.20,+0.10,-0.10}`.

11. **WHAT:** “Hard-cap each lambda ramp at 5 min” is not implemented by any cited solver API.  
    **WHY:** This is a fake mitigation unless the driver runs each ramp in a subprocess/alarm-controlled boundary.  
    **WHAT TO DO:** Either implement a real timeout wrapper or remove the claim and cap by `max_ss_steps_per_rung`.

12. **WHAT:** Sanity check #3 is tautological if `R_net` is computed as `k_des·Γ`.  
    **WHY:** It will pass even if the residual-side forward/backward source is inconsistent.  
    **WHAT TO DO:** Compare assembled diagnostics: `R_forward_capped - denominator_kprot*gamma` against `k_des*gamma`.

13. **WHAT:** The σ sign sanity check is overclaimed. “σ_S should become more cathodic as R_net grows” is not an invariant under coupled Stern, K sink, proton source, and transport feedback.  
    **WHY:** A valid physical run could fail this check.  
    **WHAT TO DO:** Treat it as an observation, not a sign-bug gate; sign bugs need manufactured-source ablations.

14. **WHAT:** The branch-ratio sanity check is also overclaimed. `x_2e` decreasing with `k_hyd` is plausible, not guaranteed.  
    **WHY:** η, transport, c_H, and K redistribution can dominate.  
    **WHAT TO DO:** Log `x_2e` and flag reversals for analysis, but do not call them bugs without decomposition evidence.

15. **WHAT:** Transport re-entry is not guarded. Baseline V_kin has `o2_flux_levich_ratio=0.631`; high `k_hyd` may push it toward or past transport limitation.  
    **WHY:** Then the k_hyd response is no longer clean kinetics/cap characterization.  
    **WHAT TO DO:** Gate or at least annotate every λ=1 rung with `o2_flux_levich_ratio` and `|cd|/I_lim_4e`.

16. **WHAT:** A.2 omits the previously deferred `dRnet_dr_H_El`/Singh sensitivity while claiming to inherit the v10a' guardrails.  
    **WHY:** v10a' found Singh amplification ≈1; A.2 should either prove that remains true across the k_hyd ramp or stop pretending it informs Singh/v10b calibration.  
    **WHAT TO DO:** Add a cheap optional `r_H_El` perturbation at selected k values, or explicitly defer it and remove v10b-Singh claims.

17. **WHAT:** The plan calls `K0_R4e_factor=1e-14` inherited, but does not state whether v10b calibration will use the same artificial R4e scale.  
    **WHY:** If v10b uses a different ORR kinetic baseline, A.2 selectivity/current outputs are not transferable.  
    **WHAT TO DO:** Add a config invariant: A.2 and v10b calibration share the same `K0_R4e_factor`, or label A.2 as only a numerical diagnostic.

18. **WHAT:** “If Γ saturates but denom_cap/total <0.8” is algebraically incoherent at λ=1. Saturated Γ implies θ near 1, and θ equals `denom_cap/total`.  
    **WHY:** This routing branch cannot mean what it says.  
    **WHAT TO DO:** Replace with a slope-based saturation test, e.g. `d log Γ / d log k_hyd < threshold`, plus independent cap ratio.

VERDICT: ISSUES_REMAIN