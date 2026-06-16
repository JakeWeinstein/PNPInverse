1. WHAT: Re point 2, the proposed helper erases sign by using `|σ_C_m2|`.  
WHY: Singh consumes a positive cathodic magnitude, but the model still needs signed σ to anode-clamp correctly. If the helper always takes absolute value, anodic bias can spuriously activate hydrolysis.  
DO: Make `sigma_C_m2_to_counts_pm2` return signed counts/pm². Then define `sigma_singh = max(0, -sigma_signed_counts_pm2)` at the pKa layer.

2. WHAT: Re point 5, the new sequence skips the σ_S(V) / V_kin instrumentation step.  
WHY: You now go `v10a → Minimum A.2`, but A.2 requires a V_kin chosen from solved σ_S(V), branch currents, and hydrolysis sensitivity. That cannot happen without a minimum V-sweep diagnostic first.  
DO: Sequence should be `v10a → minimum A.1/V-sweep diagnostics → choose V_kin → A.2`.

3. WHAT: Re point 7, `dR_net/dσ_S` is not a neutral V_kin selector unless kinetic parameters are fixed.  
WHY: The sensitivity depends on k_hyd, k_des, Γ_max, λ, cation, and local c_M/c_H. With smoke values it may select a voltage that disappears after v10b calibration.  
DO: Define the sensitivity parameter set explicitly, and add a fallback if no voltage passes the filters.

4. WHAT: Re point 9, Ring Onset Potential is an output threshold, not a potential to interpolate model cd onto.  
WHY: You must compute model ring/H2O2 current vs V and find the crossing at 0.01 mA/cm². Interpolating cd at the experimental onset potential is a different observable.  
DO: Define extraction functions per observable: threshold crossing for onset, max over fixed V-window for max ring/selectivity, and exact sign/reference conventions.

5. WHAT: Re point 9, ±30% relative tolerance is invalid for voltages.  
WHY: Relative error around 0 V or sign-changing potentials is meaningless. A 30% rule also treats selectivity, current, and voltage as if they have the same noise model.  
DO: Use observable-specific tolerances: mV for potentials, percentage points for selectivity, relative/absolute floor for currents.

6. WHAT: Re point 9, “N-1 of N observables pass” is too weak.  
WHY: It can pass while missing the primary observable, e.g. H2O2 selectivity.  
DO: Mark primary observables that must pass, then use secondary observables for support.

7. WHAT: Re point 12, porous-carbon capacitance literature is not automatically Stern C_S.  
WHY: CMK-3 supercapacitor capacitance can include internal surface area, roughness, pseudocapacitance, and normalization by mass or BET area. The model C_S is local interfacial capacitance per geometric/electrode area.  
DO: Document the area normalization and roughness mapping before importing any literature value.

8. WHAT: Re point 3, “imposed Singh σ” must not be treated as a physical coupled run.  
WHY: A per-cation constant Singh σ bypasses the solved Stern field and can make ΔpKa voltage-independent. That is useful as an algebra ablation, not as a predictive electrostatic model.  
DO: Label it as `pKa_override_ablation`; keep local Stern σ as the only coupled-physics path.

9. WHAT: Re point 6, silent Γ clamping can hide a broken Picard formula.  
WHY: The closed-form Langmuir update should naturally give `0 ≤ Γ ≤ Γ_max` for positive inputs. If it does not, clamping masks a residual inconsistency.  
DO: Clamp warm-start restores, but after Picard update assert/log if the unclamped value leaves bounds.

10. WHAT: Re point 11, `gap_Cu = 0` for Li is a rounding artifact.  
WHY: If you force Li gap to exactly zero, Singh Eq. 4 gives zero Li ΔpKa, but the deck/Singh Li shift is about −0.44.  
DO: Store high-precision back-fit values or parameterize directly in the geometric factor G.

11. WHAT: Re point 8/9, “Highest H2O2 Selectivity” requires a fixed scan window.  
WHY: A model can increase its apparent maximum by scanning a wider or different voltage range than the experiment.  
DO: Freeze the experimental V window, pH bin, cycle aggregation, and max-selection rule before comparing.

12. WHAT: Re point 14, the Phase 0 contract is still too vague if it omits the data-reduction protocol.  
WHY: The group can approve “selectivity/ring onset/local pH” but later reject how you extracted them.  
DO: Include the exact proposed metrics, tolerances, pH binning, and aggregation in the one-page contract.

13. WHAT: The revised sequence puts v10b before full plumbing ablations.  
WHY: Literature calibration before `apply_h_source/apply_k_sink/override_sigma` ablations risks fitting around a sign/plumbing bug.  
DO: Move the source/sink/override ablation matrix before v10b calibration, even if other full A.1 reporting waits.

VERDICT: ISSUES_REMAIN