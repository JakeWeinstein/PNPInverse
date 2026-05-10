Confirmed first: the parallel 2e/4e structure is real. `_bv_common.py` defines R2e/R4e with powers 2/4 and E° 0.695/1.23 V, and `forms_logc.py` uses those factors in the log-rate residual. The sequential R0/R1 framing is obsolete.

1. **WHAT:** The Phase 6α evidence basis is inconsistent. The local `iv_curve.json` files I checked record `enable_water_ionization: false` and `kw_eff_hat_target: 0.0`, despite the plan claiming post-water-ionization results.
   **WHY:** Step 3 would branch from the wrong surface pH/current fields.
   **WHAT TO DO:** Do not use the current sweep as 6α evidence until each JSON config and anchor ladder prove `Kw_eff -> KW_HAT`.

2. **WHAT:** Plan lines 100-104 conflate ionic strength `I = 0.3 M` with `[SO4]_T = 0.3 M`.
   **WHY:** For `0.1 M K2SO4`, total sulfate is `0.1 M`; ionic strength is `0.3 M`. This is a 3x concentration error and a charge bookkeeping error.
   **WHAT TO DO:** Use `[SO4]_T = 0.1 M`, `[K+]`/`[Cs+] ~= 0.2 M`, plus pH charge correction.

3. **WHAT:** “Compute implied surface c_H from fixed total sulfate + equilibrium” is underdetermined.
   **WHY:** `Ka = [H][SO4]/[HSO4]` gives speciation conditional on `c_H`; it does not determine `c_H`.
   **WHAT TO DO:** Add a conserved proton-condition/electroneutrality constraint, e.g. `E = c_H + c_HSO4 - c_OH`, or do not claim a pH shift.

4. **WHAT:** The pH-4 sulfate reservoir claim is overstated. With pKa2 ≈ 1.99, `[SO4]/[HSO4] ~= 100` at pH 4, so HSO4 is about 1%, not “both forms substantial.”
   **WHY:** Sulfate is not a strong pH 4-7 buffer. At pH 10.6, HSO4 is essentially absent locally.
   **WHAT TO DO:** Treat sulfate acid-base as weak/transport-limited above pH 4, and run pKa_eff sensitivity.

5. **WHAT:** Skipping activity correction kills quantitative claims, though not the sign that sulfate is mostly SO4.
   **WHY:** At `I = 0.3 M`, divalent sulfate nonideality can shift concentration `Ka` by about an order of magnitude.
   **WHAT TO DO:** Use activity-corrected `Kc = Ka° gamma_HSO4/(gamma_H gamma_SO4)` or bracket pKa_eff. Do not use a 30% quantitative gate.

6. **WHAT:** The spike’s `n ∈ {0.5, 1.0}` is wrong.
   **WHY:** The live solver uses H powers 2 and 4. Applying one exponent to total cd mixes channels incorrectly.
   **WHAT TO DO:** Compute R2e and R4e separately, then recombine electron currents.

7. **WHAT:** Multiplying old current by `(c_H_new/c_H_old)^n` is not a cd prediction.
   **WHY:** If pH moves 10.6 -> 6, R4e is amplified by ~1e19 at -0.4 V; O2 would deplete and the old `c_O2_surface` is invalid.
   **WHAT TO DO:** At minimum add a 1D O2 diffusion cap. Better: use this spike only to reject/accept kinetic sign, not magnitude.

8. **WHAT:** Even with O2 depletion, acid-form BV plus mass transport gives a plateau, not decay.
   **WHY:** O2 depletion limits flux; it does not make current fall with more cathodic potential in this monotone BV model.
   **WHAT TO DO:** Stop expecting sulfate acid buffering alone to create the deck peak/decay. The likely missing mechanism is pH-dependent kinetic regime or surface coverage/site blocking.

9. **WHAT:** The current “sulfate” baseline is already not ClO4 in the active sweep script. `l_eff_transport_sweep_csplus_so4.py` uses Cs+ and SO4 analytic Boltzmann entries.
   **WHY:** Phase 6β is not simply “replace ClO4 with sulfate”; it is “add HSO4/SO4 acid-base speciation to an already static SO4 multi-ion model.”
   **WHAT TO DO:** Rewrite the plan around the actual baseline.

10. **WHAT:** Branch A is basically impossible under the proposed spike.
    **WHY:** Raising `c_H` toward pH 6 strengthens acid-form current most at deep cathodic voltages, so it steepens the monotone curve instead of producing decay.
    **WHAT TO DO:** Make Branch A require a self-consistent solved peak, not a post-hoc proton multiplier.

11. **WHAT:** The 80 mV grid is not hiding the deck peak.
    **WHY:** The table is monotone from +0.55 to -0.40 V. The +0.075 to -0.083 jump is the cathodic foot, not a missed maximum.
    **WHAT TO DO:** A 25-point rerun can refine slope, but it will not rescue the current shape.

12. **WHAT:** The K2SO4 workbook is not a valid 30% substitute for the missing Cs+ deck column.
    **WHY:** K+ vs Cs+ is exactly one of the experimental variables affecting peak height/shape.
    **WHAT TO DO:** Use the K2SO4 file only for qualitative sign/window checks until the Cs workbook is found.

13. **WHAT:** Ratio dependence is not ignorable.
    **WHY:** R4e has H power 4 and R2e has H power 2, so sulfate/pH changes alter total current and selectivity differently for `1e-18` vs `1e-30`.
    **WHAT TO DO:** Run both ratios and report per-reaction contributions, not only total cd.

14. **WHAT:** The proposed 6β.1 conservation form is incomplete.
    **WHY:** `c_T = c_SO4 + c_HSO4` plus algebraic speciation is solvable, but the flux must be `J_T = J_SO4 + J_HSO4`; Poisson and Bikerman terms alone are not enough.
    **WHAT TO DO:** Use `c_T` as a primary variable with derived split, or add two NP variables plus a stiff equilibrium residual. The `c_T` DAE is the smaller change.

15. **WHAT:** The proton-condition extension has the right sign but missing flux work.
    **WHY:** `E = c_H + c_HSO4 - c_OH` is the conserved acid-base coordinate, but its residual needs `J_E = J_H + J_HSO4 - J_OH`, not just a source term.
    **WHAT TO DO:** Derive the weak form before implementation and add MMS/regression tests.

16. **WHAT:** 6δ should be split.
    **WHY:** Adding alkaline parallel reactions is a smaller, smoother change than a hard local-pH switch of BV parameters. But parallel acid+alkaline channels sum; Newton does not “pick” one unless kinetics make one negligible.
    **WHAT TO DO:** Do 6δ.1 as explicit acid + alkaline parallel channels using existing reaction-list machinery; reserve pH-gated switching/site coverage for 6δ.2.

VERDICT: ISSUES_REMAIN