1. **WHAT:** The v5 steady-state architecture is still missing the equation that changes the solution. You state that at steady state `∂c_MOH/∂t = 0`, so the proton BC reduces to Phase 6α.  
   **WHY:** If the volume `E = c_H - c_OH` equation is unchanged and the boundary flux is unchanged at steady state, Newton has no reason to move `c_H`. Hydrolysis becomes a diagnostic shadow, not active physics.  
   **WHAT TO DO:** Add a steady coupling: either a boundary surface-charge/Stern-BC modification from `M+ -> M(OH)^0`, or a real proton boundary flux/exchange term with nonzero steady turnover.

2. **WHAT:** “Poisson uses `c_M+` at `y=0`” is not meaningful as written.  
   **WHY:** A changed value at a measure-zero boundary does not alter a volume `dx` Poisson source. Boundary charge must enter as a `ds` term or through the Stern boundary condition.  
   **WHAT TO DO:** Define an areal charge correction, e.g. `σ_hydro = F * δ_OHP * (c_M+ - c_M_total)`, and put it in the Stern/Poisson boundary weak form.

3. **WHAT:** Boundary concentrations are volumetric, but boundary storage/charge is areal.  
   **WHY:** `c_MOH(boundary)` in mol/m³ cannot appear in `∂c_MOH/∂t ds` or surface charge without an OHP thickness/volume-per-area conversion.  
   **WHAT TO DO:** Introduce an explicit `δ_OHP` and consistently convert `mol/m³` to `mol/m²`.

4. **WHAT:** The “Boltzmann total cation at boundary” assumption is only first-order, not self-consistent.  
   **WHY:** The analytic Boltzmann closure gives charged `M+`, not total `M+ + M(OH)`. Once neutralization is significant, the distribution of total cation is no longer the old charged Boltzmann value.  
   **WHAT TO DO:** Either accept this as a controlled reduced model and label it, or solve a coupled boundary algebra for charged fraction and Stern field.

5. **WHAT:** The spike is not a valid Branch A/B gate.  
   **WHY:** It intentionally omits the only self-consistent feedback that can make hydrolysis affect steady state: field/Poisson/Stern response. It can screen plausibility, not decide implementation priority.  
   **WHAT TO DO:** Branch decision should wait for a minimal 6β.1 solver smoke with the boundary Stern-charge coupling active.

6. **WHAT:** `λ_hydrolysis * Ka_eff` is numerically convenient but physically odd.  
   **WHY:** Ramping `Ka` linearly from zero changes pKa logarithmically and creates extreme stiffness near small λ.  
   **WHAT TO DO:** Ramp a bounded activation on the boundary charge/equilibrium contribution, or ramp pKa/`log Ka` directly.

7. **WHAT:** The plan file on disk is internally inconsistent with the v5 summary. It still contains v3/v4 text, volume `R_buf`, `θ(y)`, Co-Zhang/IrOx misattribution, and the old finite-rate water formula.  
   **WHY:** This handoff will send the next implementer down rejected paths.  
   **WHAT TO DO:** Clean `docs/phase6b_next_steps_plan.md` before treating v5 as the canonical artifact.

8. **WHAT:** Calibration/holdout split is improved but still weak.  
   **WHY:** Using Singh parameters for K/Na/Li without tuning is honest, but those parameters may not transfer from their electrode/electrolyte conditions to CMK-3 ORR. Failure would be ambiguous.  
   **WHAT TO DO:** Record it as a predictive holdout, not a decisive falsification. If it fails, first audit Stern/OHP field mapping before rejecting hydrolysis.

9. **WHAT:** Deck-comparable priority should shift to K⁺ and `L_eff ≈ 16–26 µm`.  
   **WHY:** The confirmed deck baseline is K₂SO₄ and RDE Levich length is in that range. Cs remains useful as a high-effect smoke, but not the primary comparison.  
   **WHAT TO DO:** Run Cs smoke for sensitivity, then K⁺/SO₄²⁻ at deck-comparable transport before any claim against the deck.

10. **WHAT:** Boundary-only shadow is implementable in Firedrake only if it stays an algebraic `ds` expression.  
    **WHY:** A separate boundary function space coupled to volume unknowns is possible but adds layout, projection, and time-storage complexity for little benefit.  
    **WHAT TO DO:** Start with pure UFL boundary expressions from traced `c_H`, `phi`, and counterion closure; add a boundary DOF only if surface memory is explicitly needed.

VERDICT: ISSUES_REMAIN