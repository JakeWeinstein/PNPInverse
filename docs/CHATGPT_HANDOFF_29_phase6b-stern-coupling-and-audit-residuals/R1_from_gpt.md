1. WHAT: The Stern correction sign is wrong in the code’s convention. `C_S(φ_m - φ_s)` is the boundary flux imposed by [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:606) and documented in [nondim.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/nondim.py:63). Removing positive OHP cation charge should add `+F δ c_MOH` to that boundary-flux quantity, not `-F δ c_MOH`.
WHY: As written, the correction pushes the charge balance in the opposite direction and can validate the wrong feedback branch.
WHAT TO DO: Use `C_hat*ψ_S + δ_hat*(c_Mtotal_hat - c_Mplus_hat)` inside the `F_res -= [...] w ds` bracket, then run a sign-only test.

2. WHAT: The nondimensional units are not specified correctly. The proposed `F_nondim * δ_OHP_nondim * concentration` is suspect because the model concentrations are already nondimensional and `stern_coeff` is scaled by `F*c_ref*L`.
WHY: An extra Faraday factor or missing `L` changes the correction by orders of magnitude.
WHAT TO DO: Write both formulas explicitly: physical `σ_h = F δ c_MOH`; model `σ_hat_h = (δ/L) c_MOH_hat`.

3. WHAT: `δ_OHP = one Stokes diameter` is not a defensible convention. Bohra 2019 uses a 0.4 nm Stern width, described as slightly larger than the largest cation radius, while its SI lists solvated ion sizes as effective diameters. That is not the same as Linsey’s Stokes-radius table.
WHY: The correction is linear in `δ`; this is not a harmless naming issue.
WHAT TO DO: Treat `δ_OHP` as a tunable surface-excess thickness, probably bracketed around 0.3-0.5 nm, and cite Bohra only for the 0.4 nm convention, not as a C_S source. Source: Bohra 2019 RSC article and SI.

4. WHAT: The R5#4 neutralization estimate is flatly wrong. At pH 9.5, `[H+] = 10^-9.5 M = 3.2e-7 mol/m3`; K pKa 8.49 gives `Ka = 3.2e-6 mol/m3`, so `c_MOH/c_M+ ≈ 10`, not 0.1. That is about 91% neutralized. At Phase 6α pH 10.58 it is about 99% neutralized.
WHY: The reduced-model error is not a 10% perturbation; it is the dominant state variable.
WHAT TO DO: Stop treating the smoke target as below the Boltzmann-breakdown threshold. For K, the current v6 architecture is structurally insufficient if the target is pH 8-9. Source: Singh 2016 pKa table.

5. WHAT: `c_M_total(0) := Boltzmann c_M+(0)` is internally inconsistent. The analytic Boltzmann expression gives charged `M+`, not total `M+ + MOH0`.
WHY: If Boltzmann gives `M+`, then equilibrium implies `c_MOH = c_M+*Ka/c_H` and total increases. If total is conserved, charged `M+` is no longer Boltzmann and needs transport.
WHAT TO DO: Either promote cation chemistry to a dynamic/surface-reservoir model, or explicitly model Boltzmann `M+` plus a non-conserved neutral shadow. Do not call the Boltzmann field total cation.

6. WHAT: Stern-only coupling is not cation-hydrolysis buffering. It changes electrostatics and BV rate; it does not add the proton released by `M(H2O)+ -> MOH0 + H+`.
WHY: A smoke pass could just be capacitance throttling, not chemical pH buffering.
WHAT TO DO: Add a boundary proton exchange term if the mechanism is hydrolysis. A minimal form needs nonzero turnover, e.g. a surface reservoir with `R_hyd_s = k_hyd Γ_M+ - k_prot c_H Γ_MOH` plus adsorption/desorption/renewal. Add it as a boundary source to the `E = c_H - c_OH` residual; that does not conflict with the Phase 6α volume residual.

7. WHAT: `Ka_M_eff = f(η_local)` uses the wrong driver if `η_local` is BV overpotential. In this code `η = φ_applied - φ - E_eq` [forms_logc_muh.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:335). Cation hydrolysis pKa depends on cation size, surface charge/capacitance, and separation from the cathode, not ORR `E_eq`.
WHY: This would make the same cation have different pKa under R2e vs R4e bookkeeping.
WHAT TO DO: Drive pKa shift from `ψ_S`, Stern electric field `ψ_S/δ`, or surface charge density. Source: Singh 2016 describes pKa as charge/size/surface-charge/separation controlled.

8. WHAT: Step 6 expects pH to drop while current magnitude also drops. In acid-form BV, higher `c_H` usually increases cathodic rate unless the Stern η shift overcompensates.
WHY: The smoke criterion may be internally inconsistent.
WHAT TO DO: Add a pre-smoke sensitivity check: `Δ ln R ≈ p_H Δ ln c_H - α n Δ η`. Show the required Stern `Δη` before declaring the expected current direction.

9. WHAT: Deferring C_S sensitivity is not honest. The new mechanism is directly proportional to surface charge and inversely mediated through `C_S`; `C_S=0.10` is already labeled a convergence-pinned tunable.
WHY: A single-C_S smoke cannot distinguish hydrolysis physics from arbitrary Stern capacitance.
WHAT TO DO: Include at least one-voltage `C_S ∈ {0.05, 0.10, 0.20}` at `V_RHE=-0.40`, λ=0 and λ=1, before accepting the smoke.

10. WHAT: §5.5 overstates the sulfate-radius follow-through. The code says “Marcus, placeholder pending Linsey-deck check” at [scripts/_bv_common.py](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/_bv_common.py:594), not a verified Marcus table/page.
WHY: This papers over the audit residual.
WHAT TO DO: Reword to “named provisional Marcus value; exact source unchecked.” Low priority for 6β.1, but don’t mark it closed.

11. WHAT: §5.6 is mostly honest, but Step 6 still uses deck-current magnitude as a gate while K0_R4e/α_R4e calibration is blocked on the missing Tafel xlsx.
WHY: A plateau-magnitude failure could be calibration, not architecture.
WHAT TO DO: Make Step 6 architecture-only: convergence, λ=0 regression, pH direction, sign diagnostics, and monotone current response. Defer deck-magnitude gating until the Tafel file lands.

VERDICT: ISSUES_REMAIN