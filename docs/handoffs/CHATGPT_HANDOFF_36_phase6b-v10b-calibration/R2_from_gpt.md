1. **Your new `R_net` sign convention is wrong.**  
WHAT: D7-D1 says “R_net cathodic at V_kin (negative)” and D7-D4 talks about R_net sign flips.  
WHY: Current code defines diagnostic `R_net = k_des · Γ`, positive by construction (`phase6b_v10a_v_sweep_diagnostic.py:1202-1213`). The cation residual gets the negative sign, but `R_net` itself is the hydrolysis/proton-source scalar. This gate will fail correct runs.  
WHAT TO DO: Replace with `R_net > 0` for physical hydrolysis source, or apply cathodic/anodic sign checks only to `cd_mA_cm2` / branch currents.

2. **Point #2 response has an Eyring math error.**  
WHAT: You state ΔG_des ≈ 0.9 eV gives `k_des_phys ≈ 0.2/s ≈ 1 nondim`.  
WHY: At 298 K, `kBT/h * exp(-0.9 eV/kBT)` is about `0.004-0.006/s`, not `0.2/s`; nondim is about `0.02-0.03`, not 1.0. A nondim value near 1 corresponds closer to ~0.8 eV.  
WHAT TO DO: Recompute the Eyring table and use it consistently in §3.0, §3.3, metadata prior, and brackets.

3. **Your k_des bracket no longer covers your stated engineering prior.**  
WHAT: §3.3 says engineering prior `k_des_nondim ∈ [10^-2, 10^2]`, but D7-D3 and D7-D4 only sweep `{0.1, 1, 10}`.  
WHY: If the prior is four decades wide, the bracket evidence omits both endpoints.  
WHAT TO DO: Either narrow the prior to `[0.1, 10]` with corrected Eyring support, or include `{0.01, 0.1, 1, 10, 100}` in at least an analytic/cheap closed-form bracket.

4. **Metadata schema says every value is nondim, but `C_S` is dimensional.**  
WHAT: D1 schema comment says `"value": float # nondim`, while §3.0 says `C_S` remains dimensional F/m².  
WHY: This makes the metadata block internally false for one of the three calibrated parameters.  
WHAT TO DO: Add `units` / `is_nondim` fields, e.g. `{"value": 0.20, "units": "F/m^2"}` for `C_S`.

5. **No cross-file k_des consistency test.**  
WHAT: D3 adds solver `K_DES_NONDIM_V10B`; D4' adds script `V10B_KINETICS["k_des_nondim"]`; D8 only explicitly tests gamma factory defaults.  
WHY: Solver fallback and driver default can silently diverge. That was exactly the class of bug D3 was supposed to close.  
WHAT TO DO: Add a fast test asserting solver `K_DES_NONDIM_V10B == V10B_KINETICS["k_des_nondim"]`, plus metadata value equality.

6. **“No aliases” is overcorrected and will break existing callers.**  
WHAT: D3/D4 rename `GAMMA_MAX_HAT_SMOKE` away, but current v9/v10a tests and scripts still import it.  
WHY: You fixed provenance but risk breaking historical tests/scripts in the same PR.  
WHAT TO DO: Either update every caller found by grep, or keep a provenance-safe alias `GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10A_SMOKE` with a deprecation comment. The forbidden alias is `SMOKE = V10B`, not `SMOKE = frozen V10A`.

7. **Same alias problem for `SMOKE_KINETICS`.**  
WHAT: D4' renames `SMOKE_KINETICS` to `SMOKE_KINETICS_V10A`, but A.2 and step 6 currently import `SMOKE_KINETICS`.  
WHY: If not all importers are updated atomically, Phase B fails before any science runs.  
WHAT TO DO: Either keep `SMOKE_KINETICS = SMOKE_KINETICS_V10A` as a historical alias, or add an explicit grep/update step for all imports.

8. **Dual-mode CLI is contradictory.**  
WHAT: D4'/D19 make `--use-v10a-smoke` part of provenance, but Phase B step 7 calls it “optional convenience; defer otherwise.”  
WHY: If JSON keys depend on dual-mode behavior, the flag is not optional.  
WHAT TO DO: Decide: either v10b drivers are always v10b-only and historical reproduction uses old constants manually, or the dual-mode CLI is required and tested.

9. **D5 split is not implemented in the existing A.2 audit.**  
WHAT: The current `_convergence_audit` has hardcoded v10a baseline targets for `gamma`, `theta`, `sigma_S_C_per_m2`, and `cd_mA_cm2`.  
WHY: Your D5.SOFT deltas can still surface as audit failures unless the code is changed to separate hard convergence gates from informative deltas.  
WHAT TO DO: Add a Phase B/C step to refactor `_convergence_audit` into `hard_gates` and `soft_deltas`, and test that v10b physical movement does not fail the hard audit.

10. **σ_S monotonicity is still too hard as a pass/fail gate.**  
WHAT: D7-D1 requires strictly decreasing signed σ_S across C_S.  
WHY: The heuristic is plausible, but the coupled BV/PNP solve can perturb it; “strictly” also has no tolerance. This is a sensitivity diagnostic, not a mathematical invariant.  
WHAT TO DO: Use tolerance-based smoothness/expected-trend reporting. Escalate only on large non-smooth jumps or sign inconsistency, not tiny monotonicity violations.

11. **D7-D4 says “anchored at C_S = 0.20,” violating the locked anchor pattern.**  
WHAT: D7-D4 text says the Γ_max × k_des matrix is “anchored at C_S = 0.20.”  
WHY: The hard invariant is build anchor at 0.10, then runtime-bump to target 0.20. Direct 0.20 anchor is known fragile.  
WHAT TO DO: Rewrite D7-D4 to explicitly use the two-stage 0.10 → 0.20 pattern.

12. **D7-D4 sign-floor condition mixes unrelated quantities.**  
WHAT: “No sign flip on R_net that crosses both `|R_4e| > 1e-6` and the sign convention.”  
WHY: R_net sign stability should use an R_net magnitude floor, not R4e. Also R_net should be positive, per issue 1.  
WHAT TO DO: Use `abs(R_net) > floor` for R_net sign checks; use R4e floors only for R4e sign/branch diagnostics.

13. **The Singh `K_eq = k_hyd/k_prot` statement is still too simple.**  
WHAT: §3.3 says Singh pKa means `K_eq = k_hyd/k_prot`.  
WHY: In the implemented residual, reverse protonation is `k_prot * c_H * Γ / δ`; any equilibrium audit must carry concentration, surface-volume, δ, and nondim factors.  
WHAT TO DO: Reword as “Singh may constrain the hydrolysis/protonation balance after deriving the model-specific dimensional identity,” not a bare ratio.

14. **D7-D2 skip rule is inconsistent with D7-D4.**  
WHAT: D7-D2 says skip Γ_max bracket if a single literature value is locked, but D7-D4 always sweeps Γ_max/2 and Γmax·2.  
WHY: Not fatal, but the DoD is confusing: Γmax sensitivity is both conditional and mandatory.  
WHAT TO DO: Drop D7-D2 or define it as an expanded Γmax-only sweep beyond the always-required coupled matrix.

15. **The metadata lives in a Firedrake-importing module.**  
WHAT: D1 places `V10B_CALIBRATION_METADATA` in `Forward/bv_solver/cation_hydrolysis.py`, which imports `firedrake` at module import.  
WHY: Scripts/tests that only need metadata may now require Firedrake importability. Existing tests may tolerate this, but metadata is pure provenance and should not depend on FE imports.  
WHAT TO DO: Prefer a lightweight module, or mirror/export it through `_bv_common.py` if every consumer already sets up the Firedrake environment.

VERDICT: ISSUES_REMAIN