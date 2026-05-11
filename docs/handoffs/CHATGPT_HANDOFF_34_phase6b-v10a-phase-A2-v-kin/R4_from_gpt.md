1. **WHAT:** `selectivity_gap_pp` is still a single-V proxy, but the cited acceptance criterion is “per-cation max H2O2% in a V_RHE window.”  
   **WHY:** A.2 at fixed V_kin cannot prove agreement/disagreement with a window maximum. It can only inform v10b priority.  
   **WHAT TO DO:** Label this field `single_v_selectivity_gap_pp` and state it is advisory only, not an acceptance-bundle pass/fail.

2. **WHAT:** `classify_no_route_cause` pseudocode uses `max(r.theta)` and generic `r.picard_status_ok` over a per-k_hyd record.  
   **WHY:** This can accidentally inspect non-λ=1 rungs or inserted partial rungs. Routing should be based on the λ=1 candidate only.  
   **WHAT TO DO:** Define a helper like `lambda1_record(k_hyd_record)` and run all route/no-route gates only on that record.

3. **WHAT:** The callback side-channel is now sole source of truth, but if the callback itself raises before appending, `solve_lambda_ramp_from_warm_start` catches the exception and only records it in `result.rungs`, which A.2 intentionally ignores.  
   **WHY:** A callback bug can silently drop a rung from A.2 output.  
   **WHAT TO DO:** Wrap the callback body in its own `try/except/finally`; always append at least a snapshot with `callback_augment_error`.

4. **WHAT:** `single_iter` Picard success is defensible only because of current helper control flow, not from the emitted history itself.  
   **WHY:** Future solver edits could break that assumption.  
   **WHAT TO DO:** Add a comment in the classifier: with current helper code, `snes_converged=True` and `len(history)==1` implies the internal Picard rel test passed on the first update.

These are implementation nits, not plan blockers. The major conceptual and routing holes from R1-R3 are now closed.

VERDICT: APPROVED