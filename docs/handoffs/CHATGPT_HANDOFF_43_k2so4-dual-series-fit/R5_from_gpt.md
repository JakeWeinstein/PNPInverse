1. **Non-blocking:** pseudo-Huber needs its transition scale fixed up front.  
   **WHY:** Otherwise the robustness refit has an extra hidden tuning knob.  
   **DO:** Predeclare δ, e.g. in standardized-residual units.

2. **Non-blocking:** Δχ² = 4 profile widths should not be called confidence intervals.  
   **WHY:** You explicitly use conservative predictive σ and correlated bins, not a formal likelihood.  
   **DO:** Call them pragmatic profile widths unless you later formalize the likelihood.

3. **Non-blocking:** `git add -f` extracted data should pass whatever data-governance rule this repo uses.  
   **WHY:** This is external experimental data, not generated code.  
   **DO:** Confirm it is intended to be versioned; otherwise commit only extractor/provenance and keep CSVs ignored.

No blocking issues remain. The plan is now sufficiently disciplined to execute.

VERDICT: APPROVED