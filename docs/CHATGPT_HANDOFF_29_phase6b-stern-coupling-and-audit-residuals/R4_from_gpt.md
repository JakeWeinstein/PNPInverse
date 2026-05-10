1. WHAT: `k_des = 10^6 /s × 1/δ_OHP` is dimensionally wrong and not sourced. If `R_des = k_des Γ`, then `k_des` is strictly `1/s`. Multiplying by `1/δ` gives `1/(m s)`.
WHY: This contaminates the Γ residual and the fitted grouped parameter.
WHAT TO DO: Use `k_des [1/s]`. If no literature value is in hand, label it phenomenological and sweep it. Do not cite Bohra for MOH desorption unless Bohra actually gives it.

2. WHAT: The desorption path is physically plausible as a model closure, but not mass-conserving as written. Desorbed `MOH0` leaves the modeled system and is not tracked as neutral bulk species.
WHY: That is an open-reservoir approximation, not a conservative cation balance.
WHAT TO DO: State explicitly: “bulk electrolyte is an infinite reservoir; neutral MOH removal is an unresolved sink.” Do not claim full mass conservation unless you add neutral MOH transport.

3. WHAT: Fitting only `K_s` is not enough once `k_des` exists. At steady state,
`Γ = k_hyd c_M+ / (k_prot c_H + k_des)` and `H_source = k_des Γ`.
WHY: The source depends on at least two grouped quantities: equilibrium strength and turnover/Damköhler scale. `K_s` alone cannot control both Γ inventory and proton flux.
WHAT TO DO: Gate 4 must include `k_des` sensitivity, at least low/mid/high. Otherwise architecture pass/fail is arbitrary.

4. WHAT: The Singh-style pKa formula in v9 is not safe. It is missing a `ln(10)` conversion from energy/kBT to pKa units, and the sign is suspect: with your signed convention `σ_S < 0` at cathodic bias, your formula gives the wrong direction unless you silently use `|σ_S|`.
WHY: This can invert hydrolysis activation.
WHAT TO DO: Do not implement the guessed formula. Extract the exact equation from Singh SI, including sign, base-10 factor, capacitance/surface-charge convention, and distance dependence. The main paper only states dependence on cation charge/size and cathode charge density, not your exact formula. Source: Singh 2016 main text.

5. WHAT: `σ_S` including the hydrolysis Γ correction inside the pKa driver risks circular positive feedback with no independent bound.
WHY: Γ increases `σ`, which increases Ka, which increases Γ. Newton may converge, but to an unphysical high-Γ branch unless capacity/desorption bounds dominate.
WHAT TO DO: Add a continuation and branch diagnostic: compare pKa driven by bare Stern charge vs corrected Stern charge; fail if multiple fixed points or packing-near-1 behavior appears.

6. WHAT: The Stern correction may now double-count electrostatics. Dynamic K+ depletion already changes the Poisson field. Adding `+FΓ` assumes Γ represents missing compact-layer charged surface excess not already represented by the dynamic K+ boundary profile.
WHY: Without a control-volume derivation, the model may subtract the same positive charge twice.
WHAT TO DO: Derive Gauss balance for: volume K+, analytic SO4, neutral Γ, metal Stern charge. Decide whether Γ enters Stern as a surface-charge correction, only packing, or both.

7. WHAT: Gate 2 is still a serious feasibility risk. Dynamic K+ at cathodic bias is the counterion-accumulation analogue of dynamic ClO4- at anodic bias. Existing docs show 4sp dynamic remains the validation reference and keeps a dynamic-counterion ceiling.
WHY: The sign flip does not remove the saturated Debye-layer stiffness.
WHAT TO DO: Keep Gate 2, but add a continuation fallback plan now: z-ramp, k0-ramp, C_S ramp, and warm-start from the least cathodic nearby voltage. Treat failure as expected, not exceptional.

8. WHAT: The `h_index` role refactor is necessary but larger than stated. `forms_logc_muh.py` also has downstream IC paths that synthesize/interpret 4sp charged species using charge assumptions, not just `_resolve_mu_h_index`.
WHY: Build may pass while debye/Boltzmann IC or diagnostics mis-wire H/K.
WHAT TO DO: Audit every `z=+1`, `mu_h_idx`, and `resolve_h_index` caller, including IC and diagnostics. Add a test where H+ and K+ both exist and only H uses μ_H.

9. WHAT: A `Constant` cannot represent Γ if Γ is a Newton unknown. The v9 text says “Real element OR Constant updated at each Newton iteration”; the Constant option is not a coupled solve.
WHY: Updating a Constant outside Newton gives Picard lagging and wrong Jacobian.
WHAT TO DO: Use a real mixed-space unknown for Γ in Gate 3. Constant is acceptable only for manufactured fixed-source tests.

10. WHAT: The Γ residual form must be area-normalized carefully. A Real test function integrated over `ds` introduces electrode area; dividing by area is correct only if Γ is areal density and uniform.
WHY: Missing or double area factors will scale `k_des` and source strength.
WHAT TO DO: Define the residual as `G*((Γ-Γ_old)/dt - (R_net-k_desΓ))*area_electrode` or the area-divided equivalent, and test area invariance.

11. WHAT: The λ=0 hard-zero path must also disable Γ coupling in Stern and packing, not just set Γ initial value to zero.
WHY: A Real unknown with a zero source can still drift if the residual/Jacobian has nullspace or weak coupling.
WHAT TO DO: At λ=0 either remove Γ from W entirely for baseline tests, or pin Γ with a Dirichlet-like algebraic residual `Γ=0`.

12. WHAT: v9 is ready as a staged investigation plan, not as an approved architecture. The gates are now sane, but Gate 4 still depends on unresolved physics: desorption prior, pKa formula, and whether Γ Stern correction is double-counted.
WHY: Those are not minor implementation details; they determine the sign and magnitude of the claimed mechanism.
WHAT TO DO: Put these three items in the unresolved ledger before execution, and make Gate 4 explicitly falsification-oriented.

VERDICT: ISSUES_REMAIN