1. WHAT: The claim “slide-27 deliverable is pKa_eff, not cd” is overstated (`PHASE_6B...:33`, `:170`).  
WHY: The same deck says “ORR Metrics,” “Current density of H2O2 production,” and “Match behavior in experiment.” pKa is mechanism evidence, not the whole deliverable.  
DO: Reframe as “Phase 6β mechanistic subdeliverable: pKa_eff ordering,” and keep branch selectivity/current/local pH as downstream acceptance criteria.

2. WHAT: Slide 27 is Singh Cu pKa reproduced in the deck, not an experimental carbon validation target.  
WHY: Fitting K to Singh/Cu-derived pKa and “holding out” Cs/Na/Li mostly tests whether your Singh algebra reproduces Singh, not whether the Seitz/Mangan ORR model works.  
DO: Mark D/E as mechanistic transferability only; validate against RRDE ring/current/selectivity or IrOx local-pH data.

3. WHAT: The plan ignores direct per-cation observables already in the folder.  
WHY: `Summary Data-Error.xlsx` has pH ~4 per-cation ring onset, max ring current, and H2O2 selectivity; `CP_data.csv` has Ring OCP; Yash outputs include `current_density` and `selectivity`.  
DO: Add a minimum acceptance bundle: pKa ordering + surface pH trend + R_2e/R_4e selectivity/ring-derived H2O2 trend.

4. WHAT: Spearman ≥ 0.9 with four cations is fake precision.  
WHY: With n=4 and no ties, ρ_s is 1.0 for exact ordering and drops to 0.8 for one adjacent inversion. Your threshold means “perfect rank,” but says it indirectly.  
DO: Replace with “exact Li/Na/K/Cs ordering required,” then separately score magnitudes.

5. WHAT: The spacing ratio metric is underdefined and fragile (`ΔpKa(Cs)-ΔpKa(K))/(ΔpKa(Na)-ΔpKa(Li))`).  
WHY: Slide 27 reports pKa_near, not ΔpKa; using pKa_eff spacings gives a different ratio than using ΔpKa spacings. If the model collapses Li/Na spacing, the ratio blows up.  
DO: Predeclare whether the metric uses pKa_eff or ΔpKa; add a collapse failure rule and a vector error metric.

6. WHAT: “Read Singh first” is not a future blocker as written.  
WHY: The SI is already extracted in `docs/phase6/singh_2016_pka_formula.md`; Eq. 5 explicitly uses capacitance × total cell voltage. More reading may not resolve the local-Stern mapping.  
DO: Timebox any citation tracing, but prioritize a σ_S(V) plot and a two-convention mapping note: Singh cell-level σ vs model local Stern σ.

7. WHAT: The Singh σ mapping note overasserts equivalence.  
WHY: Singh’s σ includes total cell voltage and cation-dependent anode/OER overpotential; the model’s σ_S is local cathode Stern charge. Dropping the cell-voltage route changes both scale and cation dependence.  
DO: Treat local-Stern mapping as an assumption, not a fact; bracket results under both mappings or declare the parameter non-identifiable.

8. WHAT: Full A.1 before v10 wastes effort on unphysical v9 hydrolysis diagnostics.  
WHY: Even k_hyd=1e-3 gives Γ ≈ 0.306, about six monolayers; λ=0.25 already exceeds a one-monolayer Γ_max. Surface pH, R_forward, R_net from that regime are not physics.  
DO: Do minimal A.1 for invariant diagnostics only: σ_S(V), branch-current assembly, ladder diagnostics. Move physical hydrolysis diagnostics after a minimal Langmuir cap.

9. WHAT: v10 is underestimated.  
WHY: “1 week” includes new residual algebra, Picard update, Γ_max/k_des literature calibration, byte-equivalence, and Phase 6α compatibility. That is not a one-week production branch unless calibration is deferred.  
DO: Split v10 into v10a minimal cap with fixed Γ_max/k_des smoke values, then v10b literature calibration.

10. WHAT: The Langmuir Picard formula is not specified.  
WHY: Adding `(1 − Γ/Γ_max)` changes the closed form; using the v9 formula with a capped residual would be wrong.  
DO: Implement `Γ = λ F0_avg / [(1−λ)+λ k_des+λ k_prot<c_H>/δ + λ F0_avg/Γ_max]`, then test Γ→Γ_max as k_hyd→∞ and Γ_max→∞ recovers v9.

11. WHAT: `override_pka_sigma_S` is a unit/sign trap.  
WHY: Singh σ appears as positive counts/area; model σ_S is signed C/m²; docs also mention µC/cm². One silent unit mismatch changes ΔpKa by orders of magnitude.  
DO: Rename/split the flag into explicit units, e.g. `override_sigma_S_C_m2_signed` or `override_sigma_singh_count_pm2_mag`.

12. WHAT: `V_kin = least-saturated voltage with σ_S < 0` is not a reproducible criterion.  
WHY: Total cd can be partly plateaued while branch currents are not; there may be no voltage satisfying both kinetic-regime and σ_S<0 constraints.  
DO: Predeclare numerical gates: O2 surface fraction, cd/Levich fraction, nonzero R_2e/R_4e, σ_S sign, and fallback voltage.

13. WHAT: Branch-current diagnostics must distinguish rates from electron currents.  
WHY: R_2e and R_4e are not directly comparable without 2F vs 4F weighting, and RRDE selectivity uses disk/ring formulas, not raw rates.  
DO: Record raw R_j, electron-weighted j_2e/j_4e, gross H2O2 current, H2O2%, and n_e with N=0.224.

14. WHAT: The r_H_El transfer rules operate on a nearly singular parameter.  
WHY: Singh back-fit has r_H_El ≈ r_M-O; tiny shifts flip the geometric factor sign or explode sensitivity. The plan notices the Li risk but does not constrain it.  
DO: Parameterize the transfer in δ = r_M-O − r_H_El or directly in the geometric factor G, with sign and bounds enforced.

15. WHAT: The C_S problem is still not scoped tightly enough.  
WHY: `C_S = 10 µF/cm²` is uncited, while matching Singh Cu σ needs roughly 50 µF/cm² or a huge Stern drop. Treating C_S as a hidden knob can dominate every pKa result.  
DO: Before D/E, report σ_S(V) over the deck voltage range for C_S = 10, 50, and literature-bounded values.

16. WHAT: There is no evidence the Seitz/Mangan group accepts “pKa_eff ordering only.”  
WHY: Their deck says fit data and predict response; their data are RRDE/local-pH/selectivity heavy. This pivot may be optimizing for what the current solver can show, not what the group wants.  
DO: Send a one-page deliverable contract to the group before spending two weeks: pKa mechanism, surface pH, R_2e/R_4e selectivity, and which experimental table/slide counts as success.

VERDICT: ISSUES_REMAIN