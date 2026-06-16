1. **WHAT:** §2’s SHE algebra is correct, but the “SHE-flat” claim is overcalled. From the table, the RHE slope is about `+41.4 mV/pH`, but `R² ≈ 0.80`, not `0.95`; after SHE conversion the slope is `-17.8 mV/pH` with poor linearity.  
   **WHY:** A’s “parameter-free slope fingerprint” depends on S1 being clean. The table itself shows substantial scatter/curvature.  
   **DO:** Recompute onsets from raw curves under consistent iR/OCP conventions and report confidence intervals/leave-one-out slopes.

2. **WHAT:** “Proton-uncoupled ET fingerprint” is too strong. SHE-fixed onset is consistent with proton-uncoupled ET, but not unique. Field/pzc shifts, surface-site speciation, iR artifacts, local pH, and cation stabilization can mimic it.  
   **WHY:** You may prematurely lock onto A when the potential-axis/data-processing confounds are comparable to the signal.  
   **DO:** Treat A as leading hypothesis, not proof. Validate S1 first on raw/canonical axes.

3. **WHAT:** `slope = 59·(1−m)` is only sound if `m` means thermodynamic proton stoichiometry per electron in the formal potential. If `m` is a kinetic proton order in `rate ∝ c_H^m exp(-α n Fη/RT)`, the fixed-current RHE slope is closer to `59·[1 − m/(α n)]`.  
   **WHY:** Your inferred `m≈0.3` is convention-dependent. With `α≈0.58`, the same `−18 mV/pH` deficit implies kinetic order closer to `m≈0.18` for `n=1`.  
   **DO:** Define `m` explicitly. Use either an E_eq shift or a c_H kinetic factor, not both without deriving the combined slope.

4. **WHAT:** The aqueous `O2/O2•− ≈ −0.33 V vs SHE` reference does not match the observed SHE onsets, roughly `+0.09` to `+0.22 V`.  
   **WHY:** A literal outer-sphere aqueous superoxide onset is hundreds of mV off.  
   **DO:** Frame A as “SHE-anchored formal first ET,” not literal aqueous O2/O2•− thermodynamics, unless you add adsorption/field stabilization evidence.

5. **WHAT:** “Unfreezing the acid route gives the wrong sign” is directionally plausible but too absolute. It depends on how the acid route is written: RHE-fixed formal potential plus explicit `c_H` can double-count proton thermodynamics.  
   **WHY:** You risk rejecting real PCET contributions because of a model-convention artifact.  
   **DO:** Re-derive every elementary step in one convention: formal potential stoichiometry, local activities, BV exponent, and any explicit concentration factors.

6. **WHAT:** C, not B, owns the robust selectivity trend. B’s pKa 4.8 does not match the robust midpoint near pH 2.5–3, and disproportionation fastest near pKa would depress selectivity there.  
   **WHY:** B explains only the dubious S2′ peak, and even that has the wrong sign unless the branch is desorption-vs-reduction rather than disproportionation.  
   **DO:** Demote B to “only if raw, potential-aligned curves prove a real pH-4–5 nonmonotonicity.”

7. **WHAT:** B also conflicts with your own local-pH story. If surface pH reaches 9–10 under load, the HO2•/O2•− equilibrium is not sampling bulk pH 4.8 at the reacting interface.  
   **WHY:** A bulk-pH pKa coincidence is not mechanistic evidence if the intermediate sees local pH.  
   **DO:** Any B model must use local surface pH or specify that branching occurs after desorption in solution.

8. **WHAT:** C is under-split. Electrochemical H2O2 reduction and chemical/catalytic peroxide decomposition are not the same observable. The former adds disk current; the latter can kill ring current without the same disk-current signature.  
   **WHY:** A lumped sink can fit ring collapse while giving the wrong electron accounting.  
   **DO:** Split C into C1 disk H2O2 electroreduction and C2 chemical/catalytic loss.

9. **WHAT:** “RRDE n_e(V) distinguishes direct-4e from series-2e+2e” is false at one rotation rate. Disk/ring currents give escaped peroxide flux and total electrons, but direct 4e and series peroxide reduction enter as a sum.  
   **WHY:** This is a real identifiability failure. You cannot prove direct-vs-series partition from n_e(V) alone.  
   **DO:** Ask for rpm series, N2 H2O2 disk-reduction scans, peroxide-spike recovery, or transient collection data.

10. **WHAT:** The “two new physical params” claim is too optimistic. A has E0/k0/α coupling; C needs rate, order, potential dependence, site dependence, and possibly transport/residence terms.  
   **WHY:** The apparent parsimony is overstated.  
   **DO:** Present A+C as a minimal mechanism family, not a two-parameter model, until the exact parameterization is specified.

11. **WHAT:** The H2O2/HO2− language is misleading for pH 1.65–6.39. H2O2 pKa is far above this range, so HO2− stability is not the explanation for high-pH selectivity here.  
   **WHY:** It points to the wrong chemistry. The relevant effect is acid-favored H2O2 reduction/consumption, not meaningful HO2− abundance.  
   **DO:** Remove HO2− as a driver unless local pH actually reaches strongly alkaline values and the fraction is quantified.

12. **WHAT:** The iR/OCP caveat is not a side risk. A `0.19 V` axis discrepancy is comparable to the full pH-onset span.  
   **WHY:** It can fake or distort S1.  
   **DO:** Re-extract onset on raw E, sheet E, and physical `E − I Rs`; A only survives if the +41 mV/pH trend is robust.

13. **WHAT:** HSO4− is ranked too low for the acid-end collapse. At 0.1 M sulfate near pH 1.65–2.35, bisulfate buffering can materially set local proton availability.  
   **WHY:** It can change the apparent pH order and midpoint assigned to C.  
   **DO:** At least bound sulfate-buffer capacity before fitting C’s proton order.

14. **WHAT:** Missing gap: ring detection/collection may be pH dependent. Ring H2O2 oxidation efficiency, N calibration, and the sheet-vs-canonical selectivity convention can all distort peak selectivity.  
   **WHY:** S2′ and even plateau selectivity are fragile; raw ring current is safer than derived peak selectivity.  
   **DO:** Fit raw disk and raw ring first; use selectivity only as a derived diagnostic.

15. **WHAT:** Revised ranking should be conditional: A after axis validation; C as the robust selectivity owner; G mandatory wiring; H/D as serious confounds/refinements; E for cation cross-condition; B nearly dead unless S2′ survives; F bookkeeping, not a standalone mechanism.  
   **WHY:** Your current ranking is directionally right but too confident about A and too generous to B.  
   **DO:** First data ask: rpm series plus N2 H2O2 reduction scans. For A vs D specifically, repeat the pH-onset series after surface functional-group modification or titration; A should preserve the SHE-anchored slope, D should move or curve with site chemistry.

VERDICT: ISSUES_REMAIN