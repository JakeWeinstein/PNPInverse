1. **P21 still does not create a Firedrake-free import path.**  
WHAT: You moved the module to `Forward/calibration/v10b.py`.  
WHY: Importing `Forward.calibration.v10b` still executes `Forward/__init__.py` first. That file imports `Forward.bv_solver`, which imports the heavy BV package and Firedrake-facing modules. So the proposed test will fail for the same reason as before, just one package level higher.  
WHAT TO DO: Put the leaf module outside `Forward`, e.g. `calibration/v10b.py` with an empty top-level `calibration/__init__.py`, or make `Forward/__init__.py` lazy/lightweight, which is a broader refactor and should not be part of v10b.

2. **P25 analytic Γ_ss audit uses the wrong state data.**  
WHAT: It says to emit a 30-corner closed-form table using `F₀_avg` from the v10a'/v10b A.2 record.  
WHY: `F₀_avg`, `c_H_avg`, and therefore `B = k_prot c_H / δ` are state-dependent. Changing `k_des` and `Γ_max` can change the converged boundary state. Comparing solver corners against an analytic value computed from a baseline A.2 state can produce false “mismatches” that are just changed transport/field coupling.  
WHAT TO DO: For solver-vs-analytic consistency, compute Γ_ss using each same-corner solver rung’s own diagnostics (`forward_avg_no_k_hyd`, `c_H_avg`, `k_prot`, `delta_ohp_hat`, `k_des`, `gamma_max`). If you also want a baseline-state predictive table, label it as predictive and do not gate on it.

3. **P25 omitted the B term in the audit description.**  
WHAT: The text writes `Γ_ss = F₀ / (k_des + B + F₀/Γ_max)` but only mentions pulling `F₀_avg`; it does not specify how `B` is sourced.  
WHY: Without the current-rung `c_H_avg`, `k_prot`, and `δ_OHP`, the closed-form check is incomplete.  
WHAT TO DO: Define the audit as a call to the existing `gamma_ss_langmuir(...)` helper using same-rung diagnostics, not a hand-rebuilt partial formula.

4. **P23 grep test needs to avoid false positives.**  
WHAT: `test_v10b_production_drivers_use_V10B_kinetics` “greps” production files for `SMOKE_KINETICS`.  
WHY: The v-sweep file will intentionally contain the alias definition and historical comments. A raw grep can fail valid code.  
WHAT TO DO: Make it AST/import-aware or grep only import/use sites in A.2, step 6, and new v10b drivers. Whitelist alias-definition lines explicitly.

5. **P29 source comment uses an emoji.**  
WHAT: The planned deprecation comment starts with `⚠`.  
WHY: Source comments should stay boring ASCII unless there is a clear reason; this is not one.  
WHAT TO DO: Use `# DEPRECATED ...` without emoji.

VERDICT: ISSUES_REMAIN