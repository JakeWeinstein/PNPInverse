1. **The “lightweight” module location still imports Firedrake.**  
WHAT: P4 puts `v10b_calibration.py` under `Forward/bv_solver/`.  
WHY: Importing `Forward.bv_solver.v10b_calibration` executes `Forward/bv_solver/__init__.py`, which imports `mesh.py`, `dispatch.py`, `forms_logc_muh.py`, etc. Those pull Firedrake. So P15 is not actually fixed, and importing constants into `scripts/_bv_common.py` would make `_bv_common` FE-heavy.  
WHAT TO DO: Put the pure metadata module outside the heavy package, e.g. a new lightweight top-level package with an empty `__init__.py`, or keep duplicated constants with cross-file tests. Do not import `Forward.bv_solver.*` from `_bv_common.py` for pure constants.

2. **Your “existing flags” claim is false.**  
WHAT: Section #8 says historical reproduction can pass V10A constants via existing `--gamma-max`, `--k-des`, `--c-s` flags.  
WHY: The current v-sweep, A.2, and step 6 parsers do not expose those flags. They only expose voltage/k0/k_hyd/output/plot-ish options.  
WHAT TO DO: Either add and test those flags, or delete the historical-reproduction claim and rely only on frozen aliases/manual code paths.

3. **`SMOKE_KINETICS = V10A` can silently keep A.2/step 6 on old physics.**  
WHAT: P4 keeps `SMOKE_KINETICS = SMOKE_KINETICS_V10A`, while A.2 and step 6 currently import `SMOKE_KINETICS`.  
WHY: Unless every v10b driver import is changed to `V10B_KINETICS`, the deprecation alias preserves imports but routes v10b reruns through v10a values.  
WHAT TO DO: Add an explicit grep/update step: all v10b execution paths must import/use `V10B_KINETICS`; only historical scripts/tests may import `SMOKE_KINETICS`.

4. **No C_S consistency test.**  
WHAT: P9 cross-file consistency checks gamma and k_des, but not `C_S_F_M2_V10B` versus `STERN_F_M2_BASELINE`.  
WHY: Step 7’s locked value can drift from the driver baseline without a test catching it.  
WHAT TO DO: Add `STERN_F_M2_BASELINE == C_S_F_M2_V10B == V10B_CALIBRATION_METADATA["C_S"]["value"]`.

5. **Coupled Γ_max × k_des matrix excludes the prior endpoints.**  
WHAT: The prior is now `[0.01, 100]`, but D7-D4 couples only `{0.1, 1, 10}`.  
WHY: The hidden coupling issue was specifically about denominator compensation; endpoint k_des values can interact with Γ_max differently from the central decade. The 1D sweep does not close the coupled-risk argument.  
WHAT TO DO: Either run the full `3 × 5 × 2 = 30` matrix, or add a cheap analytic closed-form endpoint coupling check and explain why solver endpoints are unnecessary.

6. **Step 6 audit refactor is ambiguous and can weaken D6.**  
WHAT: P5 says to refactor `_convergence_audit` and “drop in `_baseline_reproduction_audit` in step 6 driver similarly.”  
WHY: D6 requires A0 byte-equivalence as a hard contract. If step 6 baseline reproduction gets softened like A.2 deltas, the plumbing-ablation contract is diluted.  
WHAT TO DO: State explicitly: A.2 v10a-delta fields are soft; step 6 A0-vs-v10b-A2 byte-equivalence remains hard.

7. **R_net sign-stability gate is now mostly vacuous.**  
WHAT: P7 says R_net sign is always non-negative and gives a sign-stability floor.  
WHY: With positive `k_des` and nonnegative Γ, R_net sign should not flip except from a bug already caught by positivity. It is not a useful sensitivity gate.  
WHAT TO DO: Replace R_net sign-stability with `R_net >= 0` sanity plus smoothness/relative-change diagnostics.

8. **R_4e sign convention still needs an explicit expected sign.**  
WHAT: P6 says “R_4e sign convention (R_4e cathodic at V_kin)” but does not say whether the diagnostic convention is positive or negative.  
WHY: Current diagnostics show branch currents positive while `cd_mA_cm2` is negative after sign conversion. An executor could check the wrong sign.  
WHAT TO DO: Define the diagnostic sign explicitly, e.g. `R_4e_current_nondim > 0` under the current observable convention at V_kin.

9. **Runtime deprecation warnings would be a mistake.**  
WHAT: R16 mentions a “deprecation warning at alias definition site.”  
WHY: If that means `warnings.warn` at import time, it will pollute tests and normal script imports.  
WHAT TO DO: Use comments/docstrings and tests, not runtime import warnings. If you need enforcement, add a grep/lint check later.

VERDICT: ISSUES_REMAIN