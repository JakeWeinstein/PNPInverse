1. **WHAT:** Re point 1 still says `pc/cd = 0.5` is branch-degenerate. With the actual signed observable, it is not. `cd = -I_SCALE*(R2e + 2R4e)` and `pc = -I_SCALE*(R2e - R4e)`. Observed `pc > 0` and `pc/|cd| ≈ 0.5` implies `R2e ≈ 0`, `R4e` dominant. **WHY:** The artifact will still confuse readers if it says both branch mixes are equally consistent. **WHAT TO DO:** Say old `pc`, if interpreted literally, points to 4e dominance, but per-branch assembly is still required because `pc` is deprecated.

2. **WHAT:** “Apparent electron count ≥ 4” is sloppy. ORR cannot exceed 4 e⁻/O₂ in this model; the ratio says “approximately 4,” with small numerical/constant mismatch. **WHY:** `≥4` suggests impossible chemistry or accounting error. **WHAT TO DO:** Use “near 4e O₂ Levich limit.”

3. **WHAT:** The sequence conflicts with itself. The ablation matrix says run A5 “after A.1” with “Langmuir cap,” but the Langmuir cap is not implemented until the later v10 capacity branch. **WHY:** That ablation cannot run in the stated phase. **WHAT TO DO:** Move A5 after v10, or redefine A5 as uncapped-current-v9 diagnostic.

4. **WHAT:** Phase D is still not a well-defined calibration. “Calibrate `r_H_El_K_carbon` at the model’s reachable σ range” against what scalar target? The Spearman/spacing metric is a cation-series Phase E metric, not a K-only calibration metric. **WHY:** You cannot fit K using a rank metric across cations. **WHAT TO DO:** Define the K scalar target: K absolute ΔpKa, normalized K shift at reachable σ, or no K fit at all.

5. **WHAT:** “Calibrate ordering” remains underdefined. Ordering is a holdout score, not a parameter-identification target. **WHY:** It will let arbitrary parameter choices pass as “calibrated” if rank is correct. **WHAT TO DO:** Separate calibration from evaluation: fit one scalar on K absolute/normalized ΔpKa, then evaluate rank and spacing on Cs/Na/Li.

6. **WHAT:** The new flags are described inconsistently as “default-on semantics” and “default-off path.” **WHY:** This matters for byte-equivalence tests. **WHAT TO DO:** State defaults exactly: `apply_h_source=True`, `apply_k_sink=True`, `override_pka_sigma_S=None`; those defaults reproduce v9.

7. **WHAT:** A4 “replace sulfate with inert background” is still not a concrete control. **WHY:** A fixed background charge, analytic inert species, and removed Bikerman species are different models. **WHAT TO DO:** Specify the replacement residual and charge/packing behavior before treating A4 as interpretable.

8. **WHAT:** The C_S-coupled transfer rule is physically ambiguous. If C_S is electrode/material-level, it should not vary by cation; if it is cation-specific interfacial capacitance, that is a new mechanism. **WHY:** This could smuggle in a per-cation fit. **WHAT TO DO:** Define whether κ is one global K-fitted scalar applied to all cations, or drop the rule from predictive Phase E.

VERDICT: ISSUES_REMAIN