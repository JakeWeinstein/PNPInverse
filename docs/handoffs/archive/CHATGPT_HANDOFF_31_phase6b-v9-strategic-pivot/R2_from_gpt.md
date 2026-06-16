1. WHAT: Point 4’s magnitude metric is dimensionally wrong.  
WHY: `Σ |error_i| ≤ 30% of mean(|deck_i|)` compares a four-cation sum to a one-cation mean, making the tolerance 4× too strict by accident. Also “argmax matches” is not exact ordering.  
DO: Use `mean_i |error_i| / mean_i |ΔpKa_deck_i| ≤ 0.30`, and state exact full ordering separately.

2. WHAT: Point 11’s unit conversion is wrong.  
WHY: `1 cm² = 1e20 pm²`, not `1e16 pm²`. For signed C/m² to counts/pm² the multiplier is `(N_A/F) * 1e-24`. A 1e4 unit error will destroy ΔpKa.  
DO: Write one tested helper for `C/m² -> counts/pm²`; do not duplicate the conversion in docs and code.

3. WHAT: Point 7’s “scale local σ_S by anode-OER overpotential” is not a defined mapping.  
WHY: Singh’s cell voltage is not a scalar correction to local Stern charge; it folds in anode kinetics, iR, Nernst terms, and their cation dependence.  
DO: Use two explicit models: local Stern σ, and Singh table σ as an empirical imposed σ. Do not invent an anode scale factor unless you build a cell-voltage model.

4. WHAT: Point 6 says re-derive σ from `|C_S ψ_S|` over expected `V_RHE`, but `ψ_S ≠ V_RHE`.  
WHY: The Stern drop is solved from the PNP/Stern split; using applied voltage as Stern voltage repeats the same scale error you are trying to audit.  
DO: Compute σ_S from solved `phi_applied - phi_surface` at each voltage and C_S.

5. WHAT: Point 8’s v10a-first reorder still needs instrumentation inside v10a.  
WHY: You cannot test the Langmuir cap without reporting `F0_avg`, `Γ`, `θ=Γ/Γ_max`, capped forward rate, and denominator terms.  
DO: Make those v10a diagnostics part of v10a, not “minimum A.1 after v10a.”

6. WHAT: Point 10’s Langmuir formula is right, but the residual safety is incomplete.  
WHY: If a stale warm-start has `Γ > Γ_max`, `(1−Γ/Γ_max)` makes the forward rate negative. That is unphysical and can destabilize Newton/Picard.  
DO: Initialize/clamp Γ into `[0, Γ_max]`, and add a test that vacancy factor never goes negative in production paths.

7. WHAT: Point 12’s V_kin gates can select the wrong voltage or no voltage.  
WHY: `cd/Levich < 0.7` may pick near-open-circuit points with weak chemistry; requiring both branches >0.01 mA/cm² may fail if the calibrated model is intentionally highly selective. The fallback “minimize cd/Levich” makes this worse.  
DO: Select V_kin by maximum predicted hydrolysis sensitivity in a non-plateau region, with branch thresholds relative to total current.

8. WHAT: Point 13 mixes intrinsic selectivity and RRDE ring conversion.  
WHY: `N=0.224` is needed to convert ring current to disk-basis H2O2 current; it is not part of model-intrinsic `R_2e/(R_2e+R_4e)` selectivity.  
DO: Report both: intrinsic selectivity from rates, and predicted RRDE observables after applying collection efficiency.

9. WHAT: Phase E validation target extraction is still underspecified.  
WHY: “Highest H2O2 Selectivity,” ring onset, and max ring current are maxima/interpolated quantities across noisy cycles and pH values. You have not defined averaging, pH matching, voltage interpolation, or error-bar use.  
DO: Freeze a data-reduction protocol before fitting: pH bin, cycle aggregation, mean/std, interpolation rule, and pass/fail tolerance.

10. WHAT: Point 1’s proposed negative-result interpretation is too strong.  
WHY: Failure to move cd/selectivity could mean missing cation-intermediate energetics, wrong sulfate/OH closure, wrong C_S mapping, wrong Γ_max/k_des, or wrong ORR kinetics. It does not uniquely imply “cation effect upstream of hydrolysis” or an H+ bottleneck story.  
DO: Phrase negative results as falsifying this model closure, not the broader cation mechanism.

11. WHAT: Point 14’s δ_HE sign convention is confused.  
WHY: You define `δ_HE = r_H_El − r_M-O`; Singh cathodic pKa lowering requires this to be ≤ 0. Then you say enforce either ≥0 or ≤0. Only one sign matches the intended mechanism.  
DO: Use `gap = r_M-O − r_H_El ≥ 0`, or parameterize the geometric factor directly.

12. WHAT: Point 15 cites “CMK-3 carbon capacitance bounds ~5-50 µF/cm²” without evidence.  
WHY: This can become another uncited calibration prior. Also C_S sweeps must re-solve the Stern/PNP problem; σ_S will not scale linearly with C_S once φ_surface changes.  
DO: Source the bounds first, then run actual solves for each C_S.

13. WHAT: Point 3 mislabels `CP_data.csv` as raw waveforms.  
WHY: The raw waveforms are in the `.mat` files; `CP_data.csv` is condensed. Confusing these will break traceability when validating local pH/Ring OCP.  
DO: Correct the artifact: CSV is summary, MAT files are raw.

14. WHAT: The deliverable contract should happen before Phase D/E design, not merely before a “2-week commitment.”  
WHY: If the group says selectivity/ring current is mandatory, D/E metrics and data reduction change immediately.  
DO: Move the one-page contract ahead of any Phase D/E metric finalization.

VERDICT: ISSUES_REMAIN