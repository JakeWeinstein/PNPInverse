1. **Reissued conditional issue: re-included bins must trigger a refit, not just rescoring.**  
   **WHAT:** Removed-bin discipline says a θ*-failed bin retried at polished θ is “re-included (final scoring re-run).”  
   **WHY:** If the data vector changes, the polished θ is no longer the optimum of the final objective. Rescoring is not enough.  
   **DO:** If any removed bin is re-included, rerun the bin-center polish and downstream FD/profile gates on the expanded mask.

2. **New issue: raw pH 4 Tafel data should gate cross-condition paper claims too.**  
   **WHAT:** The lock table says cross-condition transferability is blocked by catalyst identity only; raw Tafel xlsx gates only 7.1 feature chemistry.  
   **WHY:** Slide-15 θ* came from digitized peroxide-only data with known partition weakness. A paper-grade “one kinetic model” claim needs the exact pH 4 target, or it must remain explicitly digitization-conditional.  
   **DO:** Add raw Tafel xlsx as a blocker for paper-grade cross-condition transferability, or label that claim “digitized-slide conditional.”

3. **Minor: risk table row 6 still says “per-series reduced χ².”**  
   **WHY:** You renamed this correctly elsewhere. Leaving the old phrase invites the exact statistical interpretation you disclaimed.  
   **DO:** Change it to “per-series standardized residual score.”

4. **Minor: Huber refits need a differentiable objective.**  
   **WHY:** The workflow requires adjoint gradients; vanilla Huber has a kink.  
   **DO:** Use pseudo-Huber or explicitly handle the smooth approximation used for the sensitivity refit.

VERDICT: ISSUES_REMAIN