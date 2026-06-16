1. WHAT: Remaining blocker: Phase 5γ is still written as `j=0` only.
WHY: Pass B pure-4e and Pass D mixed need the same continuation for the active 4e channel. R4e has `E_eq=1.23` and H power 4, so it can be at least as stiff as R2e.
WHAT TO DO: Generalize anchor continuation to `active_reaction_indices`, e.g. Pass A `[0]`, Pass B `[1]`, Pass D `[0, 1]`. Ramp one at a time first; only try coupled ramp if sequential ramp fails.

2. WHAT: Your k0 ladder overshoots target: `3.2e-3 > K0_HAT_R2E ≈ 1.26e-3`.
WHY: Overshooting can fail a step that the target itself might pass.
WHAT TO DO: Generate the ladder programmatically with `min(next_k0, target_k0)` and stop exactly at target.

3. WHAT: `preconverged_anchors` needs to carry enough state, not just `ctx_snapshot`.
WHY: C+D warm-walk rebuilds a fresh context for each target and restores only `U` snapshots [grid_per_voltage.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/grid_per_voltage.py:377). It also needs correct `phi_applied`, z-scale=1, diagnostics/callback behavior, and matching mesh/function space.
WHAT TO DO: Store `{phi_applied, U_data, achieved_z=1.0, method}`. In Phase 1, build a context at that voltage, restore U, set z=1, run callback/diagnostics as if cold-converged.

4. WHAT: The highest-risk step is Phase 5α, not k0/dt continuation.
WHY: If Picard and residual still disagree on surface rate, every continuation will warm-start the wrong state.
WHAT TO DO: Instrument first: print/record Picard `R_list`, assembled residual boundary rates per unit area, `phi_o`, `psi_D`, `psi_S`, `gamma_s`, `c_s_list`, and `lambda_eff` at `+0.55` before running full C+D.

5. WHAT: Multi-ion linear-Debye Stern is an approximation, but acceptable for this recovery path.
WHY: You are using it as an IC/Picard seed, not changing the residual. Exact BKSA multi-ion Stern is out of scope.
WHAT TO DO: Do not add bracket machinery for the multi-ion linear shortcut. Instead assert finite positive `lambda_eff`, finite `psi_D`, and `0 <= abs(psi_D) <= abs(full_drop)`.

6. WHAT: Probability estimate: I’d put Pass A at ~70%, Pass D at ~55%, Pass B at ~40% within 3 focused days.
WHY: A is directly targeted by Picard+k0+dt. D adds coupled competition but should benefit. B has stronger 4e stiffness and placeholder kinetics, so less predictable.
WHAT TO DO: Define success as “any A or D ≥15/25 with nonzero R2e” and I’d call the whole plan ~65% likely. All of A/B/D ≥15/25 is closer to ~35-45%.

7. WHAT: Early failure detector should be one scalar: residual-rate/Picard-rate mismatch.
WHY: If mismatch is >1-2 orders after Phase 5α, the rest is wasted compute.
WHAT TO DO: Before any grid run, require `abs(log10(|R_res|/|R_picard|)) < 0.05` for non-tiny rates, or abs error floor for tiny rates.

8. WHAT: Minor implementation nit: `grid_per_voltage.py` currently imports only `set_initial_conditions_logc` directly.
WHY: Phase 5β direct branch needs `set_initial_conditions_logc_muh` too.
WHAT TO DO: Add the import or avoid direct imports by cloning params to `linear_phi` and using dispatcher.

VERDICT: ISSUES_REMAIN