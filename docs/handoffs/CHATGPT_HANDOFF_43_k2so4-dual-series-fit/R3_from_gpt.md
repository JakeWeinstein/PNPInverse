1. **Reissued: solver-failure mask is still too permissive.**  
   **WHAT:** “Persistent single-V failures at θ* ... are removed from the mask BEFORE fitting.”  
   **WHY:** That still lets numerical robustness decide the data vector. A hard-to-solve voltage may be exactly where the model is strained. Calling the post-mask grid “13/13” hides the original omission.  
   **DO:** If any in-window experimental bin is removed solely because θ* failed, paper-grade claims are blocked unless the polished θ is retried on that bin and either included or scientifically excluded. Report original bin count, removed bins, and final retry outcome.

2. **Pushback on R2#5: c_O₂ bracket is too narrow if gas is genuinely unknown.**  
   **WHAT:** If O₂ protocol is unconfirmed, refits only at `{1.0, 1.3}` mol/m³ around 1.2.  
   **WHY:** That brackets O₂-saturated solubility uncertainty, not gas identity. Air-saturated O₂ is far lower. The data probably rules air out by the plateau, but the plan should prove that rather than silently assume it.  
   **DO:** Add an air-saturated sanity/refit or a documented impossibility check against the measured plateau. Otherwise make O₂ protocol confirmation blocking for transport/kinetic claims.

3. **New: paper-grade lock is blocked only on the 0.47 V OCP, but other asks also gate paper claims.**  
   **WHY:** rpm/L_eff, O₂ protocol, catalyst identity, and ring hold/calibration all affect the interpretation. Conditional labels help, but “paper-grade lock” should not sound unlocked with those unresolved.  
   **DO:** Define lock tiers: computational lock vs paper-grade kinetic/transferability claim. List which asks block which claim.

4. **Pushback on R2#8: derived-quantity profiles are not yet rigorous enough.**  
   **WHAT:** Partition profiles use checkpoint lower envelopes plus 3–5 quadratic-penalty constrained refits.  
   **WHY:** Checkpoint clouds are not profiles, and penalty constraints can miss the true constrained optimum or understate widths. Three points may not bracket a confidence threshold.  
   **DO:** State a constraint tolerance for fixed-g refits, require enough g values to cross the chosen Δχ² threshold on both sides, and use checkpoint envelopes only as exploratory plots.

5. **Reissued FD concern: the tolerance still collapses near a stationary point.**  
   **WHAT:** `max(0.05|g_fd|, 1e-3 max_k|g_k|)` can go to numerical zero when all gradients are small.  
   **WHY:** Final polished θ should have small gradients, so FD noise can dominate.  
   **DO:** Add an objective-scale absolute floor, and run the main adjoint/FD validation at a nonstationary interior point; use the final θ check as a regression smoke test.

6. **New: the n_e diagnostic now misses low-ring, high-disk regimes.**  
   **WHAT:** It only evaluates where `I_r > 10σ_floor_ring`.  
   **WHY:** A 4e-dominated region can have meaningful disk current and near-zero ring current; excluding it weakens the side-current diagnostic.  
   **DO:** Evaluate n_e wherever disk current clears threshold, with ring uncertainty propagated. Use ring-thresholding only to flag low-confidence peroxide fractions.

7. **Minor but fix it: “per-series reduced χ²” conflicts with the declared σ framing.**  
   **WHY:** You explicitly say σ is a conservative predictive-error scale and absolute reduced χ² is not interpreted. Calling it reduced χ² invites exactly that interpretation.  
   **DO:** Rename to per-series standardized residual score unless you actually want reduced-χ² semantics.

VERDICT: ISSUES_REMAIN