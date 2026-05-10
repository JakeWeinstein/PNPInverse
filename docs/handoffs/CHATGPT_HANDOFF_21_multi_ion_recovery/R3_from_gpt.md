1. WHAT: Your Phase 5α no-Stern eta rule is wrong. You wrote `eta_drop = phi_applied - phi_o` for no Stern.
WHY: Current residual semantics use `eta_raw = phi_applied_func - E_eq_const` when `use_stern=False` and `use_eta_in_bv=True` [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:298). Current Picard matches that with `eta_drop = phi_applied_model` [picard_ic.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py:1115).
WHAT TO DO: In `_update_electrostatics`, no-Stern multi-ion must use `eta_drop = phi_applied_model`, not subtract `phi_o`.

2. WHAT: You flipped Stern physics again in point 12. Large finite `C_S` gives `psi_D -> full_drop`, `psi_S -> 0`; small `C_S` gives `psi_D -> 0`, `psi_S -> full_drop`. The code formula says this directly [picard_ic.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py:263).
WHY: Your conclusion “large finite C_S → Stern absorbs all” is backwards.
WHAT TO DO: If you use Stern homotopy, large positive `C_S` is the easy Robin-branch start; ramp down to `0.10`. It is not first priority, but it is not invalid for the reason you gave.

3. WHAT: `_update_electrostatics` is the right top-level helper, but it should compose smaller helpers.
WHY: One giant helper returning seven values will become hard to test and easy to break for byte-equivalence.
WHAT TO DO: Implement `_solve_phi_o`, `_solve_stern_split_for_picard`, `_compute_gamma_s`, then one `_update_electrostatics` wrapper that preserves current single-ion order.

4. WHAT: The monotonicity test as written will fail or mislead once you add per-ion `phi_clamp`. A clamped residual can have zero derivative outside the clamp.
WHY: `assert dρ/dφ < 0 everywhere on [-50,50]` is too strong if the closure is deliberately flat near clamp.
WHAT TO DO: Test strict monotonicity inside the unclamped operating interval, and separately assert the Picard root stays comfortably inside clamp.

5. WHAT: Your local bracket `(phi_o_prev - 5, phi_o_prev + 5)` is fine only with a fallback, but do not make global bisection the silent fallback.
WHY: If local bracketing fails, that is signal about a large Picard jump.
WHAT TO DO: Log/return a diagnostic when local bracket fails; then use global bracket only after confirming monotonicity and root-inside-clamp.

6. WHAT: Phase 5γ still underspecifies IC ordering. You must update k0 before calling the Picard IC if you want the initial IC to match small k0.
WHY: If you build forms, set IC at target k0, then assign k0 down, Picard and residual are inconsistent for the first solve.
WHAT TO DO: In the anchor builder: build context, build forms, assign k0 down in both metadata and `bv_k0_funcs`, then call `set_initial_conditions`.

7. WHAT: “Picard re-runs with new k0” during continuation is not true unless you explicitly call the IC again.
WHY: Once you are warm-starting from the previous converged solution, Picard metadata is mostly irrelevant; the residual sees `bv_k0_funcs`.
WHAT TO DO: Rename `update_picard_k0` to something like `set_reaction_k0_model`; use it before IC for consistency and during continuation for residual/diagnostic consistency, but do not overwrite warm states by rerunning IC at every k0 step.

8. WHAT: The key is `ctx["dt_const"]`, not `ctx["dt_constant"]` [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:581).
WHY: The proposed anchor builder will throw or silently miss the dt control.
WHAT TO DO: Mutate `ctx["dt_const"].assign(...)`.

9. WHAT: Existing C+D cannot consume your external anchor. If no cold successes exist, it returns immediately [grid_per_voltage.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/grid_per_voltage.py:487).
WHY: Returning a converged anchor `ctx` from `solve_anchor_with_continuation` does not feed warm-walk unless you add an injection API.
WHAT TO DO: Add `preconverged_anchors` / `initial_snapshots` support to C+D, or write a separate warm-walk routine that starts from the external anchor.

10. WHAT: k0-first vs dt-first: neither alone. Your first state should use both small k0 and small dt.
WHY: k0 attacks the BV source; dt attacks Newton step size/stiffness. They solve different parts of the failure.
WHAT TO DO: Start with `k0 small + dt small`; ramp k0 to target while keeping dt small; then ramp dt to production. Interleave only if the k0 ramp stalls.

11. WHAT: The z=0 rate explanation is closer now, but `phi(0)=phi_applied` under Stern at z=0 is not guaranteed. The Robin solve determines surface phi.
WHY: Your exact `exp(33.9)` number may be wrong, although the qualitative “bulk H plus large BV rate” point stands.
WHAT TO DO: Measure `phi_surface` and assembled `bv_rate_exprs[0]` at the z=0 fallback IC after the Phase 5β fix.

12. WHAT: Rate-consistency tolerance should not be a blind `rel_tol=1e-3`.
WHY: Scalar Picard vs UFL assembly can differ through interpolation, quadrature, area normalization, and near-zero rates.
WHAT TO DO: For constant boundary-state tests, use `1e-3` relative plus absolute floor `1e-10`. For full spatial IC tests, start at `1e-2` relative. Always divide assembled rate by electrode area.

13. WHAT: Regenerating the `logc_muh` baseline is a yellow flag, not routine cleanup.
WHY: The 15/15 sweep is load-bearing; changing fallback semantics can improve correctness while shifting legacy results.
WHAT TO DO: Keep old and new baselines side by side. Require: direct debye path unchanged where fallback is not used; fallback-only cases documented as intentional behavior change.

14. WHAT: Phase 5β direct formulation dispatch is acceptable, but use params, not `sp.solver_options`, unless you are certain `sp` is always a `SolverParams`.
WHY: `grid_per_voltage` supports both `SolverParams` and legacy 11-tuples.
WHAT TO DO: Reuse the existing `params` dict and `_params_with_phi(...)`, or write a helper that handles both representations.

VERDICT: ISSUES_REMAIN