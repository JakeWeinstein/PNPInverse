Recommendation: patch the multi-ion Picard and the `logc_muh` fallback first, then use a positive-`k0_R2e` continuation with smaller `dt_init`, then warm-walk voltage from the highest V. That has the best signal-to-hours ratio. I would not spend first effort on 5b I-ramp, 5g, or softplus.

1. WHAT: The reviewed JSON does not match the handoff. [anchor_smoke.json](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/StudyResults/fast_realignment_2026-05-08/anchor_smoke/anchor_smoke.json:2) says `anchor_v_rhe = 0.0`, not `+0.55`, and its fields differ from the quoted dump.
WHY: You are ranking recovery paths from mixed evidence.
WHAT TO DO: Save separate `+0.55` and `0.0` artifacts; do not use this JSON as the `+0.55` diagnostic.

2. WHAT: The `R_2e ≈ 0.45` estimate is incomplete. The residual uses `u_exprs` in log-rate [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:431), and the multi-ion IC applies `log_gamma` to O2 and H [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:1092). Also H transport is limiting through `_surface_concs_from_rates` [picard_ic.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py:907).
WHY: The true SS is probably not simply `c_O = 0.55`; H and H2O2 surface states matter.
WHAT TO DO: Recompute the target using the Picard flux balance, not a bulk-H boundary-rate estimate.

3. WHAT: Multi-ion Picard is still wrong. It uses single-ion gamma [picard_ic.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py:1138), single-ion `phi_o = log(H_o/c_clo4_bulk)` [picard_ic.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py:1212), and single-ion Stern split [picard_ic.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py:1228).
WHY: The post-Picard override fixes only the spatial IC potential; it leaves `O_s/P_s/H_o/R` seeded from the wrong kinetics.
WHAT TO DO: Patch Picard before continuation. But patch `phi_o`, `lambda_eff`, `psi_D`, per-ion outer concentrations, and `gamma_s` together.

4. WHAT: The `logc_muh` fallback is invalid. The orchestrator calls `set_initial_conditions_logc(...)` directly [grid_per_voltage.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/grid_per_voltage.py:350) instead of the dispatcher.
WHY: In `logc_muh`, species 2 is `mu_H`, not `u_H`; the fallback corrupts the proton variable and contaminates the z=0 failure.
WHAT TO DO: Replace line 350 with dispatched `set_initial_conditions(ctx, sp)` or a `logc_muh`-aware fallback.

5. WHAT: Even the `logc_muh` linear fallback is single-ion. It uses first counterion bulk and `phi_o = log(H_b/c_clo4_bulk)` [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:662).
WHY: With Cs first, that gives the same wrong `-7.6` bulk anchor.
WHAT TO DO: Add a multi-ion linear-Stern path using `solve_outer_phi_multiion()` and `effective_debye_length_local()`.

6. WHAT: 5b I-ramp is not the best first move. Counterion bulk concentrations are baked into UFL constants in the Boltzmann closure [boltzmann.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/boltzmann.py:220), so every α needs form rebuild and state transfer.
WHY: It is more engineering than k0 continuation and does not address the corrupted fallback or wrong Picard seed.
WHAT TO DO: Try I-ramp only after the correctness patches and k0/dt continuation fail; keep electroneutrality during the ramp.

7. WHAT: 5g targets the wrong variable first. The proposed better `phi_outer(y)` still depends on wrong Picard `c_s` if Picard is not patched.
WHY: The observed miss is boundary-rate and surface-concentration consistency, not proven spatial interpolation error.
WHAT TO DO: Defer 5g until a rate-consistent Picard IC still fails and diagnostics show EDL-shape jumps.

8. WHAT: Softplus/tanh rate bounding changes the problem. A per-reaction cap also ignores shared O2 and H transport between R2e and R4e.
WHY: It may converge a modified residual with the wrong selectivity/current, and it conflicts with the production trust model around clip=100 [clipping_conventions.md](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/clipping_conventions.md:8).
WHAT TO DO: Do not use it as production physics. Only use as a homotopy with final unsaturated residual verification.

9. WHAT: “Transient pre-relax” is already what `run_ss` does: repeated implicit solves with the time term [grid_per_voltage.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/grid_per_voltage.py:260).
WHY: Same `dt_init=0.25` will hit the same first-step Newton failure.
WHAT TO DO: Make this concrete as `dt_init` continuation, e.g. `1e-4 → 1e-3 → ...`, with higher step caps.

10. WHAT: The Stern continuation direction is backwards. Smaller Stern capacitance gives smaller `psi_D`; the linear formula is `psi_D ∝ stern_coeff / (eps + stern_coeff*lambda)` [picard_ic.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py:263).
WHY: Starting at `C_S=0.01` worsens H depletion protection, not improves it.
WHAT TO DO: If used, start from larger `C_S` or no-Stern, then ramp down to `0.10`.

11. WHAT: k0 continuation is viable, but exact zero is not. Disabled reactions are hardwired to `R_j = 0` at form build [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:407).
WHY: Assigning `ctx["bv_k0_funcs"][j]` later cannot re-enable a skipped branch.
WHAT TO DO: Build with tiny positive `k0_R2e`, solve, then geometrically assign up to target. Do this for R2e, not only R4e.

12. WHAT: The “start at low V_RHE” alternative has the sign wrong. The page grid’s `+0.55` is already the weakest cathodic drive [peroxide_window_3sp_parallel_2e_4e_csplus_so4.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/studies/peroxide_window_3sp_parallel_2e_4e_csplus_so4.py:44).
WHY: Walking from `-0.5` starts from a harder ORR point, not an easier anodic one.
WHAT TO DO: Anchor at the highest V, or use an artificial above-Eeq homotopy if allowed.

13. WHAT: The legacy-ClO4 IC swap is cheap but low-trust. It gives `phi ≈ 11`, tiny H, and near-bulk O/H2O2, while the target multi-ion residual wants small diffuse drop and sulfate/Cs charge balance.
WHY: The multi-ion Poisson residual may reject that state before BV benefits from the low-H rate.
WHAT TO DO: Keep it as a one-hour falsification after the Picard/fallback patches, not as the primary path.

14. WHAT: Multi-ion counterion diagnostics are wrong. `diagnostics.py` computes `theta_b` per counterion [diagnostics.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/diagnostics.py:195), not shared across Cs and sulfate.
WHY: Saturation and counterion surface means in failure JSON are not reliable.
WHAT TO DO: Reuse `build_counterion_ctx()` and `per_ion_outer_concs()` for diagnostics.

VERDICT: ISSUES_REMAIN