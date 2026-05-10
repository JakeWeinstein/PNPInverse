1. WHAT: Your Issue 8 defense picks the wrong fast-path steric ion. You claim SO4²⁻ is repelled at `V_RHE > 0`, but the code’s Boltzmann sign gives `c_SO4 = c_b exp(+2φ)`. Positive `φ` attracts sulfate.
WHY: At the proposed `+0.55 V` anchor, ideal sulfate can explode first. This recreates the exact steric blow-up ClO4 Bikerman was added to avoid, now with `z=-2`.
WHAT TO DO: Either make sulfate the single steric ion for the positive-voltage anchor, or implement both-ion sterics. Cs-steric/SO4-ideal is not a safe default.

2. WHAT: `_solve_outer_phi_multiion(H_o, c_b_list, z_list)` is written as ideal electroneutrality, but your proposed electrolyte is Cs steric + SO4 ideal.
WHY: `H_o + c_Cs exp(-φ_o) - 2 c_SO4 exp(2φ_o)=0` is not the same model as the residual if Cs uses Bikerman saturation.
WHAT TO DO: The φ_o solve must call the same concentration closure as Poisson: dynamic H/O/P packing, Cs steric expression, sulfate ideal expression.

3. WHAT: `compute_surface_gamma_multiion` uses bulk `c_b` anchors, but once `H_o` changes, the analytic ions’ outer concentrations are not bulk concentrations.
WHY: The diffuse-layer γ should be anchored at the outer state, not the reservoir state. Otherwise Picard rates, φ_o, and the spatial IC are mutually inconsistent.
WHAT TO DO: Solve φ_o first, compute outer ion anchors from φ_o, then use those anchors in γ. If sulfate is ideal, `a_SO4` must be zero in γ.

4. WHAT: Your Issue 15 “maybe fix `poisson_coefficient`” is dangerous.
WHY: `poisson_coefficient` is the base nondim coefficient using `C_SCALE`; the residual’s linearized Debye length comes from `poisson_coefficient / Σ z_i² c_i`. Changing the coefficient to ionic strength would double-count screening in the PDE.
WHAT TO DO: Leave the residual coefficient alone. Fix only IC/Stern helpers to use `λ_eff = sqrt(poisson_coefficient / Σ z_i² c_outer_i)`.

5. WHAT: “Disable Stern at IC only” is not justified, and the claim that Stern is small at 0.55 nm is backwards.
WHY: At 0.3 M, diffuse capacitance is roughly ε/λ_D ≈ 1.3 F/m²; with Stern `0.10 F/m²`, the Stern layer can dominate the voltage drop.
WHAT TO DO: Implement a linearized multi-ion Stern split for the IC, or ramp Stern capacitance during continuation. Do not simply omit it and call the mismatch bounded.

6. WHAT: Disabled-reaction semantics are inconsistent with the pure-channel passes.
WHY: Issue 6 says topology detection should ignore disabled reactions; Issue 17 needs pure-2e and pure-4e to exercise the two-reaction parallel plumbing. If disabled reactions are ignored, pure-4e becomes a one-reaction topology, not a parallel-topology probe.
WHAT TO DO: Classify topology from the nominal two-reaction config, but assemble disabled reactions as exact zero rows/zero rates.

7. WHAT: Phase 2.1 only updates Picard helpers, but the spatial IC interpolation still hardcodes `ln(H_outer / c_clo4_bulk)` and single-ion `gamma_psi` in both form backends.
WHY: Picard can be mathematically fixed while the actual FE seed remains ClO4-specific and poisons Newton.
WHAT TO DO: Add an explicit multi-ion spatial IC path in both `forms_logc.py` and `forms_logc_muh.py`; remove `c_clo4_bulk` from that path entirely.

8. WHAT: The adapters still set `c_clo4_bulk = counterions[0]["c_bulk_nondim"]`.
WHY: With `[Cs, SO4]`, that variable becomes Cs; with `[SO4, Cs]`, it becomes sulfate. Either way it is not a ClO4 anchor and should not feed φ_o, Stern, γ, or IC construction.
WHAT TO DO: Pass a structured counterion list/context into Picard and IC code. Do not overload `c_clo4_bulk`.

9. WHAT: The “done” criterion was lowered from `15/25` to `10/25` without user approval.
WHY: The user asked for an end-to-end page-15 sweep. `10/25` is a partial diagnostic, not a sweep.
WHAT TO DO: Keep done at `≥15/25` for the mixed real-topology stack, or explicitly ask the user to accept a lower bar.

10. WHAT: Pass A pure-2e is now the only pass required to show non-trivial gross H2O2, but pure-2e is not the real experimental topology.
WHY: The mixed pass with placeholder `k0_4e` still makes R2e invisible; the plan can “pass” while producing no meaningful peroxide curve under the actual parallel setup.
WHAT TO DO: Add a mixed-topology continuation/bracket pass with reduced `k0_4e` that preserves both channels and yields inspectable R2e, clearly separate from the uncalibrated physical-placeholder run.

11. WHAT: The legacy warm-start fallback is underspecified.
WHY: You cannot “switch” an existing ctx from ClO4/sequential to Cs/SO4/parallel; only the compatible `U` state can be copied into a freshly built target ctx.
WHAT TO DO: State the exact operation: solve legacy ctx, build target ctx separately, assign matching `U_prev`/`U` subfunctions for O2/H2O2/H/φ, then solve target residual.

VERDICT: ISSUES_REMAIN