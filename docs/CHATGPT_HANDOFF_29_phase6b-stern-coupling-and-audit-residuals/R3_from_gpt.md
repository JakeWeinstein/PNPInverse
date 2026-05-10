1. WHAT: v8 still has no nonzero steady proton source if the only Γ equation is reversible hydrolysis. With `R_f = k_hyd c_M+` and `R_r = k_prot c_H Γ_MOH`, the natural surface balance is `∂t Γ_MOH = R_f - R_r`. At steady state, `R_f = R_r`, so net H+ production is zero.
WHY: This recreates the v7 steady-state-null problem, just with Γ as a DOF.
WHAT TO DO: Add an actual turnover/removal path, e.g. `∂tΓ_MOH = R_f - R_r - R_des + R_ads`, with proton source `R_f - R_r`. Without desorption/exchange, hydrolysis only transiently buffers.

2. WHAT: The boundary mass balance prose is chemically wrong. “M+ flux in = ∂tΓ_MOH + R_prot_release” confuses reverse protonation with product removal. Reverse protonation consumes H+ and converts `MOH0 -> M+`; it is not proton release.
WHY: The signs for M+ sink, Γ growth, and H+ source will be wired wrong.
WHAT TO DO: Define one net reaction rate `R_net = R_f - R_r`. Then use it consistently: M+ boundary sink = `R_net`, Γ production = `R_net` minus removal, H+ source = `R_net`.

3. WHAT: The `k_prot` units are wrong. For `R_r = k_prot c_H Γ_MOH`, units require `k_prot` = `m3/(mol s)`, not `m4/(mol s)`.
WHY: This breaks the Ka mapping and nondim scaling.
WHAT TO DO: Use `Γ_eq = δ_OHP * Ka_eff * c_M+ / c_H`. The deck pKa can transfer as a volume-equivalent Ka if you consistently interpret `Γ/δ` as the local MOH concentration.

4. WHAT: `logc_muh` cannot currently support dynamic K+ plus H+. It hard-fails when more than one species has `z=+1` in [_resolve_mu_h_index](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:87), and water ionization has the same exactly-one-`z=+1` assumption in [water_ionization.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/water_ionization.py:79).
WHY: v8 cannot even build with H+ and K+ both dynamic.
WHAT TO DO: Add explicit species roles: `h_index`, `mu_species=[h_index]`, and do not infer proton by charge alone.

5. WHAT: Dynamic K+ at cathodic bias is not obviously easier than dynamic ClO4- at anodic bias. It is the same “counterion accumulates into a saturated Debye layer” stiffness with the sign flipped.
WHY: The known 4sp dynamic ceiling is not safely irrelevant; v8 may fail before hydrolysis.
WHAT TO DO: First smoke must be λ=0, no Γ, dynamic K+ + analytic SO4, `V_RHE=-0.40`, clip=100, C+D. No hydrolysis work should start until that converges.

6. WHAT: The λ=0 regression cannot be byte-equivalent to Phase 6α. The DOF layout changes, K+ moves from analytic Boltzmann to dynamic NP, and φ/H are coupled to that new residual.
WHY: “Byte-equivalent original-DOF subset” is an impossible gate.
WHAT TO DO: Replace with semantic tolerances: original observables and residual blocks match Phase 6α within specified norms after the dynamic K+ equilibrium solve.

7. WHAT: Γ_MOH as “facet-supported Function” is not current solver plumbing. Existing `R_space` is for global constants, not unknowns, and the mixed space is only volume CG components in [forms_logc.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc.py:70).
WHY: This is a real Firedrake/mixed-space refactor, not a small helper module.
WHAT TO DO: For the 1D/RDE-uniform electrode smoke, use one global `R` scalar Γ coupled through `ds`. Defer spatial facet Γ until there is evidence it is needed.

8. WHAT: The Stern correction should now be written in Γ units, not `δ*c_MOH` units. If `Γ_hat = Γ/(C_ref L)`, then the nondim Stern correction is simply `+Γ_hat`.
WHY: Mixing `c_MOH_hat`, `Γ_MOH`, and `δ` invites another unit bug. For fixed Γ, the Stern charge is independent of arbitrary δ.
WHAT TO DO: Store Γ with an explicit nondim scale and use `σ_hat_h = Γ_hat`; use `Γ/δ` only for packing and equilibrium.

9. WHAT: The hybrid “dynamic K+ + analytic SO4” closure is not the same as the current two-analytic multi-ion closure. The IC, shared-theta packing, Poisson source, and diagnostics all need a new consistency proof.
WHY: A subtle double-count or wrong denominator will dominate the smoke.
WHAT TO DO: Add a λ=0 manufactured/equilibrium test: dynamic K+ + analytic SO4 must reproduce the old analytic K/SO4 charge density and packing on a fixed φ profile.

10. WHAT: The calibration problem is now underdetermined. `k_hyd`, `k_prot`, Γ capacity/removal, `δ`, `β_M`, `C_S`, plus K0_R4e/α_R4e cannot all be learned from LSV/CP data, especially with the Tafel xlsx missing.
WHY: v8 can fit anything and explain nothing.
WHAT TO DO: Freeze most parameters from literature or independent priors. Fit only one or two grouped parameters at 6β.1, such as `K_s = δ Ka_eff` and one turnover/removal rate.

11. WHAT: The pKa/field model is still too hand-wavy. The Singh-style term should use field or surface charge with dielectric and closest-approach assumptions stated. `β_M` cannot absorb all of that and remain transferable.
WHY: Otherwise cation-series validation is meaningless.
WHAT TO DO: Write the exact formula before implementation, including units, sign, dielectric, radius, and whether `σ_S` includes the hydrolysis correction.

12. WHAT: The smoke plan is still too ambitious for a pivot this large.
WHY: You have at least three independent feasibility risks: dynamic K+ build/convergence, Γ mixed-space implementation, and hydrolysis mass-balance identifiability.
WHAT TO DO: Split 6β.1 into gates: build/form test for H+ plus K+, λ=0 dynamic-K equilibrium smoke, Γ-only manufactured source test, then finite hydrolysis at one voltage.

VERDICT: ISSUES_REMAIN