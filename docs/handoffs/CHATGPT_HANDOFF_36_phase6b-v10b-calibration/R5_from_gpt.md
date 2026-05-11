1. **P31 calls `gamma_ss_langmuir` with the wrong keyword names.**  
WHAT: The patch shows `gamma_ss_langmuir(lam=..., delta=..., F_avg=...)`.  
WHY: The actual helper signature is `lambda_val`, `delta_ohp`, and `forward_avg`. Executing the plan literally raises `TypeError`.  
WHAT TO DO: Specify the real call:
```python
gamma_ss_langmuir(
    lambda_val=1.0,
    k_hyd=k_hyd_rung,
    k_prot=k_prot,
    k_des=k_des_rung,
    delta_ohp=delta_ohp_hat,
    forward_avg=forward_avg_no_k_hyd,
    c_H_avg=c_H_avg,
    gamma_max=gamma_max_rung,
)
```

2. **P31 still has an ambiguous/dangerous `F_avg` definition.**  
WHAT: It says the solver emits `F_avg`, but current diagnostics distinguish `F0_avg` from `forward_avg_no_k_hyd`.  
WHY: `gamma_ss_langmuir` multiplies `k_hyd * forward_avg` internally. If the driver passes `F0_avg`, it double-counts `k_hyd`.  
WHAT TO DO: Require the per-rung record to use `forward_avg_no_k_hyd` for the helper, or add a wrapper that accepts `F0_avg` and does not multiply by `k_hyd`.

3. **Analytic Γ mismatch should not be merely soft.**  
WHAT: P31 says mismatch `>1e-3` is a soft writeup flag, not escalation.  
WHY: This is not a cosmetic trend diagnostic. It checks whether the reported solver Γ matches the same closed-form update used by the model. If it fails, the matrix evidence is suspect.  
WHAT TO DO: Make this a D7-D4 hard gate, or add a separate hard mass-balance residual gate for every matrix rung and explicitly say the analytic table is only advisory.

VERDICT: ISSUES_REMAIN