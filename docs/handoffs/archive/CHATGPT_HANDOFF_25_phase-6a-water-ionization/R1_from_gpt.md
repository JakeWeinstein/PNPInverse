1. **§4 reason 3 has wrong kinetics arithmetic and reverses the timescale conclusion.**  
   `k_r·c_H_bulk ≈ 1.4e-4 s^-1` is dimensionally/arithmetically wrong. With `k_r = 1.4e11 M^-1 s^-1` and `c_H_bulk = 1e-4 M`, it is `1.4e7 s^-1`. At `c_H = 1e-9 M`, it is `1.4e2 s^-1`, not `1.4e-8 s^-1`. Also `τ_ss ≈ L_eff²/D_H ~ 1 ms at 100 µm` is wrong: `(1e-4)^2 / 9.3e-9 ≈ 1.1 s`.  
   **Why it matters:** the stated evidence for fast-water is numerically incoherent. The conclusion may still be true, but the proof in the plan is broken.  
   **Fix:** rewrite the Damköhler section using consistent M or mol/m³ units and compare `1/(k_r(c_H+c_OH))` against `L_eff²/D`.

2. **§3 Option D Damköhler is wrong by orders of magnitude.**  
   The plan says `Da = k_r · L²/D ~ 10^9`, omitting concentration. The correct scale is `k_r c_typ L²/D`. Depending on `c_typ`, this ranges widely.  
   **Why it matters:** this exaggerates stiffness and biases the option ranking.  
   **Fix:** compute `Da_H = k_r c_H L²/D_H` and `Da_OH = k_r c_OH L²/D_OH` over the expected pH range and L_eff sweep.

3. **§2 / §4 claim “water sources fresh H+” is physically misleading under Option B.**  
   Algebraic `c_OH = Kw/c_H` does not add a source term to the H+ transport equation. It only adds OH- charge and steric occupancy to Poisson/Bikerman. H+ is still consumed by BV and replenished only by H+ flux from the boundary.  
   **Why it matters:** Option B may not raise surface `c_H` at all except indirectly through electrostatic feedback. The main acceptance target, `surface pH < 9`, is not guaranteed by the proposed implementation.  
   **Fix:** explicitly state that Option B enforces water equilibrium only in the charge model, not in H+ mass balance. If the goal is proton replenishment, you need Option C or a dynamic/source formulation.

4. **§5 Q2 missing H+ source/sink is a blocker.**  
   Adding `-Kw exp(-u_H)` to Poisson is not equivalent to water self-ionization. Water equilibrium imposes both `c_H c_OH = Kw` and mass/charge transport constraints.  
   **Why it matters:** the solver can still drive `u_H` to extreme depletion while simply creating algebraic OH-. That may worsen electroneutrality stress instead of fixing pH.  
   **Fix:** either downgrade Option B to “electrostatic OH- closure only” or add a conservation-respecting variable such as proton condition/alkalinity.

5. **§3 Option B violates OH- conservation and charge-current consistency.**  
   Slaved OH- can appear/disappear pointwise as `c_H` changes, but there is no OH- flux and no reaction term balancing the H+ equation.  
   **Why it matters:** Poisson sees negative charge that did not enter through any flux or reaction balance. This can distort potential, Stern drop, and BV rates nonphysically.  
   **Fix:** add diagnostics for integrated charge neutrality and current balance, or use Option C/D for production claims.

6. **§3 Option C is underspecified and probably not as simple as stated.**  
   `E = c_H - c_OH` cancels water reaction, but the flux is `J_E = J_H - J_OH`, with different diffusivities and opposite charges. You cannot solve a scalar diffusion equation for `E` without constitutive closure through `c_H(E)`, `c_OH(E)`, and `φ`.  
   **Why it matters:** the fallback may be more invasive than the plan admits, especially in log variables and BV boundary fluxes.  
   **Fix:** write the actual PDE: `∂E/∂t + ∇·(J_H(E,φ) - J_OH(E,φ)) = electrode/source terms`, with boundary conditions.

7. **§3 Option C has units hidden in the quadratic formula.**  
   `c_H = (E + sqrt(E² + 4Kw))/2` is valid only if `Kw` is in the same concentration-squared units as `E²`. The plan mixes M, mol/m³, and nondim throughout.  
   **Why it matters:** one missed conversion gives pH errors of 6 orders of magnitude.  
   **Fix:** define `E_hat`, `Kw_hat`, and write the nondim formula explicitly.

8. **§5 Q1 `KW_NONDIM = KW_M2 * (1000.0 / C_SCALE) ** 2` is correct only if `KW_M2` means M², but the name is dangerous.**  
   `M2` looks like square meters, not molar squared.  
   **Why it matters:** this is exactly the kind of constant that causes silent 10^6 errors.  
   **Fix:** rename to `KW_MOLAR2 = 1e-14`; add test asserting pH 4 bulk gives `c_OH = 1e-10 M = 1e-7 mol/m³`.

9. **§5 Q1 `c_bulk_nondim = KW_NONDIM / C_HP_HAT` assumes `C_HP_HAT` exists and is nondim H+ bulk.**  
   If `C_HP_HAT = 0.1/1.2`, then fine. If it is physical mol/m³ or absent in the config path, this silently breaks.  
   **Why it matters:** config constants are already a known source of fragile plumbing.  
   **Fix:** compute from one canonical physical `C_H_BULK_M3` and `C_SCALE`, then test the value.

10. **§3 Option A sign critique is incomplete.**  
   For an anion, Boltzmann is usually `c = c_bulk exp(-z φ)` in nondim electrostatic potential. With `z=-1`, that is `exp(+φ)`. If cathode `φ < 0`, OH- decreases. That part is fine. But the plan ignores that outer electroneutral pH response is not an EDL Boltzmann problem.  
   **Why it matters:** the rejection is right for the wrong scope: inert Boltzmann may still be valid inside an EDL for a fixed bulk OH-, but not as a bulk water-equilibrium model.  
   **Fix:** distinguish EDL equilibrium from homogeneous reaction equilibrium.

11. **§5 Q3 IC formula is not obviously valid for slaved OH-.**  
   The proposed `a_OH · Kw/H_outer · (e^ψ - 1)` assumes OH- varies Boltzmann-like across ψ while also being slaved to H+. But if H+ itself is dynamic/logc and water equilibrium holds, `c_OH = Kw/c_H`; whether that becomes `Kw/H_outer * e^ψ` depends on the sign convention and on whether `c_H` follows `H_outer e^{-ψ}` in the IC.  
   **Why it matters:** one sign error in the IC can seed enormous OH- in the wrong region and kill Newton.  
   **Fix:** derive the IC equations from `μ_H`/`u_H` conventions used in code, then add a direct numerical test: initialized `c_H*c_OH/Kw ≈ 1` at every node.

12. **§5 Q2 says `forms_logc.py` / `forms_logc_muh.py` need no changes. That is probably false.**  
   The water-ionization expression needs access to the H+ primary variable and, in muh, to `φ`. Existing Boltzmann expression builders may only receive `φ` and static config.  
   **Why it matters:** the touch surface is larger than claimed.  
   **Fix:** inspect the actual function signatures and list required data dependencies explicitly before implementation.

13. **§5 Q2 missing derivative/Jacobian discussion.**  
   `Kw exp(-u_H)` contributes `-Kw exp(-u_H)` to derivatives with respect to `u_H`. In muh it also depends on `φ`.  
   **Why it matters:** if the expression is not included symbolically in UFL correctly, Newton either sees the wrong Jacobian or a detached expression.  
   **Fix:** verify the residual uses UFL expressions tied to the current unknowns, not precomputed arrays.

14. **§6 R2 clamp is wrong-sided/incomplete.**  
   Existing `u_clamp = 100` for `exp(u_H)` does not necessarily protect `exp(-u_H)` when `u_H` is very negative.  
   **Why it matters:** cathodic depletion is exactly where `exp(-u_H)` explodes.  
   **Fix:** introduce an explicit lower clamp for `u_H` or upper clamp for `c_OH`, and test pH 14-16 regimes.

15. **§6 R3 underestimates OH- concentration.**  
   At pH 14, OH- is 1 M, not `~1e-5 M`. At pH 9, OH- is `1e-5 M`. The current failed model reports pH near 14, so slaving initially implies huge OH-.  
   **Why it matters:** OH- steric packing and charge density may absolutely dominate near the cathode if you apply `Kw/c_H` to the existing depleted solution.  
   **Fix:** recompute worst-case OH- from the failed sweep surface pH and include a clamp or continuation strategy.

16. **§5 acceptance criterion 2 has the sign wrong conceptually.**  
   “Fresh H+ source raises the supply ceiling” should make the cathodic plateau more negative in magnitude, but from `-0.089` toward deck `-0.18` is not a “lift” unless you define lift as magnitude.  
   **Why it matters:** ambiguous score logic leads to wrong pass/fail interpretation.  
   **Fix:** state: “plateau magnitude increases; cd becomes more negative.”

17. **§5 validation P3 alone is insufficient.**  
   You can force `pH < 9` with a clamp or electrostatic artifact while breaking current conservation or charge neutrality.  
   **Why it matters:** the model could pass the headline criterion for the wrong reason.  
   **Fix:** add acceptance checks for integrated current balance, electroneutrality away from EDL, `c_H*c_OH/Kw`, and comparison to Yash OH- profiles.

18. **§1 / §2 pH and concentration numbers are inconsistent.**  
   The text says `c_H` crashes to `~10^-9 mol/m³`, which is `10^-12 M`, pH 12. The reported pH 13.72 corresponds to roughly `1.9e-14 M = 1.9e-11 mol/m³`.  
   **Why it matters:** the order-of-magnitude argument is using the wrong depletion regime.  
   **Fix:** standardize every pH statement with both M and mol/m³.

19. **§7 “sulfate buffering weak at pH 4” is too glib.**  
   HSO4-/SO4²- has pKa around 2, so mostly dissociated at pH 4, but at 0.1 M sulfate even a small buffer fraction can matter compared with 0.1 mM H+.  
   **Why it matters:** buffer capacity may exceed water autoionization for pH excursions below/near neutral.  
   **Fix:** quantify buffer capacity before deferring it.

20. **§5 tests include “byte-identical output” after adding new optional config. That is brittle.**  
   Floating output can change from import order, metadata, or harmless config serialization.  
   **Why it matters:** this creates noisy failures unrelated to physics.  
   **Fix:** use numerical regression tolerances and assert the disabled path does not alter residual expressions/configured species.

21. **§5 MMS proposal is not meaningful for Option B as written.**  
   Since OH- has no PDE, “manufactured solution with `c_H c_OH = Kw`” only tests algebraic reconstruction, not transport correctness.  
   **Why it matters:** it gives false confidence.  
   **Fix:** for Option B, unit-test algebraic residual contributions. Save MMS for Option C/D where OH-/E has an actual PDE.

22. **§4 “Option B vs Option C is second-order from OH- flux divergence” is unsupported and likely false.**  
   Near pH 9-14, OH- gradients can be enormous and `D_OH` is large. Flux divergence is not automatically second-order.  
   **Why it matters:** this is the central approximation. If wrong, Option B is not a first pass; it is a misleading model.  
   **Fix:** estimate `|∇·J_OH|` from Yash snapshots or from `Kw/c_H` profiles before claiming smallness.

23. **§4 escalation logic is backwards for the stated goal.**  
   If the required missing physics is proton replenishment by water, Option B does not implement that. Option C is the first conservative fast-reaction model.  
   **Why it matters:** you may spend a phase validating a shortcut that cannot satisfy the physical diagnosis.  
   **Fix:** make Option C the recommended first production implementation, or relabel Option B as a diagnostic experiment only.

24. **§5 Q1 `D_OH` is added but unused in Option B.**  
   **Why it matters:** including it implies OH- transport exists when it does not.  
   **Fix:** omit it from Option B config or reserve it explicitly for Option C/D.

25. **§5 Q3 bulk θ term `a_OH · Kw / c_H_bulk` must use nondim concentrations.**  
   The formula is written dimensionally ambiguous.  
   **Why it matters:** steric terms require volume fraction; using M instead of nondim/physical molar density gives wrong θ by powers of 1000/C_SCALE.  
   **Fix:** write `theta += a_hat_OH * c_hat_OH`, with `c_hat_OH = Kw_hat / c_hat_H`.

VERDICT: ISSUES_REMAIN