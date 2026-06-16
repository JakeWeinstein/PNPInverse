1. **WHAT:** Re point 3 is still wrong on branch algebra. In the code, `current_density = -I_SCALE*(R_2e + 2R_4e)` and deprecated `pc = -I_SCALE*(R_2e - R_4e)`. Observed `pc ≈ 0.5*|cd|` implies `R_2e ≈ 0`, `R_4e` dominant, not R_2e dominant. **WHY:** Your latest counterreply flips the branch conclusion again. **WHAT TO DO:** State: old `pc` is deprecated, but if interpreted literally it points to 4e dominance. Then wait for per-branch assembly before making any selectivity claim.

2. **WHAT:** The updated TL;DR says the plateau could be “4e or 2e Levich.” No. The observed total current is already near the **maximum 4e O₂ electron-current ceiling** for the O₂ flux. Pure 2e O₂ Levich would be ~2.75 mA/cm² with code constants, not 5.53. **WHY:** Per-branch instrumentation can determine branch mix, but the total current magnitude already excludes a pure 2e plateau. **WHAT TO DO:** Say “O₂ Levich with apparent electron count near 4 unless per-branch assembly proves another source/accounting issue.”

3. **WHAT:** Re point 1 option B, “calibrate ordering, not absolutes,” is not a calibration target. Ordering is qualitative and likely already baked into Singh Eq. (4) under many parameter choices. **WHY:** You cannot fit or validate a scalar parameter against rank order alone without a quantitative metric. **WHAT TO DO:** Define a metric: rank only as a pass/fail holdout, plus quantitative ratios, ΔpKa spacing, or normalized shifts.

4. **WHAT:** “Our model cannot reach Singh’s Cu σ under any V_RHE” is too strong until you instrument the actual σ range. Also Singh’s σ comes from total cell-voltage/capacitance bookkeeping, not necessarily the same object as your local Stern charge. **WHY:** You may be rejecting absolute pKa matching because of a mapping mismatch, not a true solver exclusion. **WHAT TO DO:** First plot model `σ_S(V)` and document the Singh-to-Stern mapping assumption.

5. **WHAT:** Re point 5 has the k_des direction muddled. With a Langmuir cap, maximum net source is `k_des*Γ_max`; if that is too small, **larger** `k_des` increases net source, smaller `k_des` decreases it. **WHY:** The proposed literature interpretation is backwards. **WHAT TO DO:** Treat `k_des`, `Γ_max`, and max acid flux as a coupled physical calibration.

6. **WHAT:** “Punt on the cap and live with architectural debt” is not acceptable for production if current converged Γ is ~64 monolayers. **WHY:** That makes the existing high-Γ regime physically invalid, not merely incomplete. **WHAT TO DO:** Keep current v9 only as a numerical diagnostic branch; do not use uncapped Γ for physical conclusions above the monolayer threshold.

7. **WHAT:** Re point 6’s “C_S-coupled per cation” rule is a per-cation refit unless the C_S transformation is fixed from K before seeing holdouts. **WHY:** Otherwise Phase E stops being predictive. **WHAT TO DO:** Predeclare one K-fitted transformation rule per candidate family, then apply it unchanged to Cs/Na/Li.

8. **WHAT:** Running three transferability rules and reporting whichever works invites selection bias. **WHY:** This turns Phase E into model shopping unless criteria are locked. **WHAT TO DO:** Rank the rules before running, or report all three as exploratory and reserve “validated” for a future blinded criterion.

9. **WHAT:** Re point 12 says v9 architecture stays unchanged, but A.1 adds new residual flags, `override_sigma_S`, and an AdaptiveLadder behavioral patch. **WHY:** These are code-path changes, even if default-off. **WHAT TO DO:** Label them as default-off instrumentation/control changes and add λ=0/disabled-path regression tests after each.

10. **WHAT:** `manufactured_R_inj = 1e-3` may not produce a robust surface-c_H shift at the chosen voltage. **WHY:** A plumbing regression should be deterministic, not dependent on guessed magnitude. **WHAT TO DO:** Bracket `R_inj` once, then set the test value to one that reliably produces a ≥5% surface-c_H shift while converging.

11. **WHAT:** `override_sigma_S` must be scoped precisely. If it only drives `ΔpKa`, say so; if it also touches Stern/Poisson, it changes electrostatics. **WHY:** Otherwise ablation results will be uninterpretable. **WHAT TO DO:** Name it `override_pka_sigma_S` and keep it out of the Stern residual.

12. **WHAT:** The proposed `1e-4` mass-balance tolerance is premature. **WHY:** Boundary flux reconstruction in `logc_muh` plus water-ionization closure may not close at that tolerance even when the residual is correct. **WHAT TO DO:** Establish tolerance empirically on λ=0 and manufactured-source cases before making it a gate.

VERDICT: ISSUES_REMAIN