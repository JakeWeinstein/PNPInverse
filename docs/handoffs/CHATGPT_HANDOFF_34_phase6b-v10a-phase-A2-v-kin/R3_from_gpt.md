1. **WHAT:** R3-#4 still mishandles `len(gamma_picard_history) == 8`. You classify every length-8 history as `iter_cap_hit`, even if the final internal Picard rel was <1e-4 on the 8th allowed iteration.  
   **WHY:** Valid rungs can be falsely rejected. This matters most at high `k_hyd`, exactly where convergence margin is unknown.  
   **WHAT TO DO:** If `len >= 2`, compute `last_rel`; classify `len == 8 AND last_rel < 1e-4` as `converged_at_iter_cap`, not failure. Also make `snes_converged=False` override all Picard statuses.

2. **WHAT:** R3 routing dropped the independent saturation-slope criterion. `k_hyd_route` only requires `θ > 0.9`, which can be upper-knee, not plateau.  
   **WHY:** The selectivity gap may be evaluated before true cap saturation, so the bridge to v10b is biased by remaining `k_hyd` dependence.  
   **WHAT TO DO:** Add a local slope gate to `k_hyd_route`: require `d ln Γ / d ln k_hyd < 0.05` or use a stricter coverage threshold like `θ > 0.95` plus a slope report.

3. **WHAT:** `selectivity_gap_pp = deck_target_pct − observed` is undefined because the deck target is a band (`~25-50%`), not a scalar.  
   **WHY:** High/low v10b priority changes depending on whether you use 25, 37.5, 50, or nearest-bound distance.  
   **WHAT TO DO:** Define interval gap: `0` if observed is inside `[25,50]`; below band use `25 - observed`; above band use `50 - observed`.

4. **WHAT:** The callback sketch creates two possible sources of truth: it mutates `rung_diag` in place and also appends `dict(rung_diag)` to `augmented_rungs`. Since `result.rungs` holds the same dict object, both may contain augmented copies.  
   **WHY:** Easy to double-count rungs or diverge between `result.rungs` and `augmented_rungs`.  
   **WHAT TO DO:** Pick one source. Prefer the callback side-channel for partial failures and, on success, build output only from `augmented_rungs`.

5. **WHAT:** `exception_phase` is promised but no mechanism is specified. `LadderExhausted` does not carry a structured phase enum.  
   **WHY:** Implementers will end up string-matching exception messages, which is brittle.  
   **WHAT TO DO:** Either explicitly parse the three known message prefixes, or add a tiny wrapper enum in the driver: before calling the helper, assume `warm_or_lambda0`; if callback fires, phase is `lambda_positive`; if exception text contains `λ=0 floor`, classify `lambda_zero`.

6. **WHAT:** “If no `k_hyd_route`, reroute to V_kin re-selection or smaller `K0_R4e_factor`” is too prescriptive. No route could also mean Picard ambiguity, mass-balance failure, or grid not dense enough near a clean saturated point.  
   **WHY:** You may re-run expensive V selection when the actual fix is local to A.2.  
   **WHAT TO DO:** Split inconclusive causes: `transport_only`, `picard_failure`, `mass_balance_failure`, `no_saturated_clean_rung`, `grid_gap`. Route each separately.

7. **WHAT:** The transition criterion still says “span θ ∈ [0.10, 0.93]” but does not define the test precisely.  
   **WHY:** “Span” can mean all points inside the interval, min/max bracket the interval, or at least one point near each end. Those produce different pass/fail outcomes.  
   **WHAT TO DO:** State exact logic, e.g. `min(theta_converged) <= 0.10 AND max(theta_converged) >= 0.93 AND at least 4 transition-grid points converged at λ=1`.

VERDICT: ISSUES_REMAIN