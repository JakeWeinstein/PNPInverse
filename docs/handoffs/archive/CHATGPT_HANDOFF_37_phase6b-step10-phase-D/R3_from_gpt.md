1. WHAT: v3 still has a T-grid contradiction. It says `T=0` is “baseline” in #3, then correctly says in #5 that `T=0` maps to the sign-flip boundary `Δ_β≈+45.6`, not baseline. WHY: this will produce off-by-one/invalid-domain bugs in the grid builder. DO: remove `T=0` from evaluated candidates; keep it only as the open upper bound. Explicitly include `Δ_β=0` with its actual `T≈-4.88e-6`.

2. WHAT: The optimizer bound is open at `+45.6`, but `minimize_scalar(method="bounded")` only supports closed finite bounds. WHY: it can evaluate arbitrarily close to or at a sign-invalid boundary depending on tolerance/numerics. DO: use `upper = -β_K_Cu - eps`, with `eps` defined in ΔpKa space, e.g. `eps_beta = 1e-6 / σ_ref`.

3. WHAT: The Stern lower bound still comes from `T=-5`, but the v3 grid only goes to `T=-3`. WHY: bracket, grid, and optimizer domain no longer match. DO: either include `T=-5` in the grid or make the lower bound correspond to `T=-3`.

4. WHAT: The ablation target grid is wrong. At `override_sigma=0.141`, `Δ_β=0` already gives `T≈-6.43`, but the v3 grid only spans `0` to `-3`. WHY: the ablation optimizer would exclude the existing Cu-prior baseline and all stronger-hydrolysis candidates. DO: build the ablation bracket separately and force inclusion of `Δ_β=0`; extend ablation T coverage at least through the baseline and safe lower domain.

5. WHAT: The same T grid cannot be blindly reused for Stern and ablation. WHY: in Stern coordinates, `Δ_β=0` is near zero effect; in ablation coordinates, it is a large effect. DO: define mapping-specific grids in ΔpKa space, each with explicit `Δ_β=0`, sign boundary, and finite safe lower bound.

6. WHAT: Pre-screening uses `σ_local_max_across_V_grid` from a `Δ_β=0` baseline, but D34 pins only a single-V byte-equivalence check. WHY: a single `V_kin` check cannot provide max σ over the grid. DO: run a full `Δ_β=0` scan for pre-screen σ bounds, while separately comparing the `V_kin` record to A.2.

7. WHAT: Even a full `Δ_β=0` σ bound may underpredict σ at large negative `Δ_β`. WHY: the residual changes H/K profiles and can change Stern charge. DO: after each successful solve, verify assembled per-V `|pka_shift_avg| <= 15`; if violated, mark the candidate invalid and exclude it from finite loss.

8. WHAT: v3 says invalid-domain candidates “contribute +inf” and also says identifiability uses finite non-invalid candidates only. WHY: those are different metrics. DO: define two arrays: `loss_all` with invalid markers for traceability, and `loss_finite_valid` for identifiability/optimization.

9. WHAT: The V grid section still says “22 points total” before listing 24 values. WHY: this kind of mismatch already caused two rounds of grid bugs. DO: remove the stale count and make the constant plus unit test the source of truth.

10. WHAT: `V=-0.10` is outside the locked overlap window `[−0.06,+1.0]`. WHY: if extraction includes `-0.10`, max selectivity/ring current can be computed outside the deck overlap. DO: keep `-0.10` only for `V_kin` sign/baseline diagnostics; mask acceptance observable extraction to `[−0.06,+1.0]`.

11. WHAT: The continuation order is internally inconsistent: “DESCENDING” but `+0.55 → +1.0` is ascending. WHY: solve order matters for convergence and reproducibility. DO: list the exact solve order, including the off-grid anchor `+0.55`, the anodic branch, and the cathodic branch.

12. WHAT: The anchor voltage `+0.55` is not in the explicit grid, which has `0.54` and `0.59`. WHY: it is unclear whether `+0.55` is only an anchor or also an emitted data point. DO: state that anchor-only points are excluded from observable aggregation, or add `+0.55` to the grid.

13. WHAT: `compute_beta_per_cation()` in `calibration/v10b.py` risks a circular import. `scripts/_bv_common.py` already imports `GAMMA_MAX_HAT_V10B` from `calibration.v10b`; if `v10b.py` imports `SINGH_2016_CATION_PARAMS` from `_bv_common`, it will cycle. WHY: this can break Firedrake-free imports and existing v10b tests. DO: put the helper and Singh table in a separate Firedrake-free module, e.g. `calibration/singh2016.py`, and import it from both places.

14. WHAT: The pinned A.2 comparison field names do not match the actual JSON. Existing keys are `cd_mA_cm2`, `theta`, `gamma_final`/`gamma`, not `cd_mA_cm²`, `θ`, or `gamma_solver`. WHY: the byte-equivalence test will fail for schema reasons before testing physics. DO: pin the exact JSON path: `per_k_hyd_records` row with `k_hyd_target==1e-3`, final `lambda=1` rung, exact key names.

15. WHAT: The baseline comparison does not specify how to choose the `k_hyd=1e-3` record from `phase_a2_v_kin.json`. WHY: that file contains multiple `per_k_hyd_records`; blindly using first/last is wrong. DO: select by `k_hyd_target == 1e-3` and `lambda_hydrolysis == 1.0`, then assert uniqueness.

16. WHAT: The wall-budget arithmetic is still inconsistent: it says 37 evals, then 47, then 31-41. WHY: the plan cannot be scheduled or reviewed if the count is unstable inside one paragraph. DO: give one table: baseline, Stern grid, Stern optimize, ablation grid, ablation optimize, final checks, total.

17. WHAT: The ablation-Brent decision is still unresolved in text: “reuse pre-fit only” then “still need Brent.” WHY: σ-divergence requires an actual ablation fit if following the bundle. DO: either commit to ablation Brent or explicitly downgrade to grid-best ablation with rationale.

18. WHAT: `xatol_beta = 0.05 / σ_local(V_kin)` assumes the loss-relevant σ is V_kin σ. WHY: max selectivity may occur at another voltage with different σ. DO: set tolerance using `max σ_clamped` over the acceptance window or optimize directly in target-ΔpKa coordinates with a mapping wrapper.

VERDICT: ISSUES_REMAIN