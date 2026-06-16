Directionally, yes: demoting A to a frame/gauge question and promoting the vertical ring response is the right move. But one blocking contradiction remains.

1. **WHAT:** pH 2 is still both “untouched held-out” and the “primary C gate.”  
   **WHY:** If the digitized pH-2 ring curve is used to choose C1/C2 vs direct-4e vs escape vs ring-efficiency, it is model selection, not held-out prediction. The plan cannot claim a genuine pH-2 stress test after using pH 2 as G2.  
   **DO:** Pick one protocol:
   - Train/select on pH 4 plus pH 6.39 anchor, then open pH 2 once as the final stress test with no iteration afterward.
   - Or use pH 2 as the lead C gate and drop the “untouched held-out” claim entirely.

2. **WHAT:** N1a needs an anchored sign-explicit voltage mapping.  
   **WHY:** `ΔV = 0.0592·pH` contains an arbitrary constant and is easy to sign-flip. The null should preserve the pH-6.39 anchor.  
   **DO:** Write the exact map, e.g. `V_model = V_reported - 0.0592*(pH - 6.39)` or its signed equivalent, then unit-test that pH 6.39 is unchanged.

3. **WHAT:** “Establishes pH-dependent peroxide yield/loss as real new physics” is still slightly too strong before the competitor test.  
   **WHY:** Your own ring-efficiency null could explain a vertical ring collapse without disk-side peroxide yield/loss.  
   **DO:** Phrase it as “tests whether pH-dependent peroxide yield/loss is required.” Claim it only if the ring-efficiency/vertical-scale null loses.

Fix #1 and the plan is basically executable. #2 is a sign-control cleanup; #3 is wording discipline.

VERDICT: ISSUES_REMAIN