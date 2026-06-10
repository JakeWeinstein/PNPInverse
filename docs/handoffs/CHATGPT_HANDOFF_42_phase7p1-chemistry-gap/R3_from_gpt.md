1. WHAT: Re 6/B′ risks creating “HSO4-laundered H+” through the existing analytic sulfate reservoir.

WHY: If `c_SO4` is an analytic counterion field with no finite sulfur/bisulfate transport constraint, `c_HSO4 = c_SO4*c_H/Ka2` can act like an infinite local buffer. That would be the same structural error as Kw-laundered H+, just with sulfate.

WHAT TO DO: Enforce finite total sulfur and buffer transport explicitly. Add sulfur-flux and HSO4-deprotonation ledgers, and require no-flux sulfur conservation in a closed test.

2. WHAT: Re 6 does not state the `Ka2` unit convention.

WHY: pKa is normally in mol/L, while the code appears to use mol/m³. Missing the ×1000 conversion changes HSO4 by 1000×. Your predicted shift is only ~0.03 V, so this is fatal if wrong.

WHAT TO DO: Hard-code/test `Ka2 = 10^-pKa M = 10.2 mol/m³` for pKa 1.99, or document an equivalent nondimensional standard. Unit test bulk pH 4, total sulfate 0.1 M gives ~1.0 mol/m³ HSO4.

3. WHAT: Re 6 does not fully specify bulk electroneutrality after the sulfate split.

WHY: Adding HSO4 changes charge balance. If Cs+, SO4²⁻, HSO4⁻, H+, and OH− bulk chemical potentials are inconsistent, Poisson will absorb a fake background charge and the pH transition can move for the wrong reason.

WHAT TO DO: Add a bulk speciation test satisfying `[Cs+] + [H+] = 2[SO4²−] + [HSO4−] + [OH−]` at the film edge, with zero field and zero residual in the uniform state.

4. WHAT: Re 6 states the storage condition `E = c_H + c_HSO4 − c_OH`, but the fast-equilibrium flux algebra is still underspecified.

WHY: The pH transition is controlled by fluxes, not just storage. HSO4⁻ and SO4²⁻ have different charges, mobilities, and steric terms. A wrong Jacobian/flux can pass byte-equivalence-off tests but still implement the wrong buffer physics.

WHAT TO DO: Write the exact residual and Jacobian formulas before coding. Add tests for buffer capacity `dE/dpH`, no-field diffusion limits, migration sign, and sulfur mass conservation.

5. WHAT: B′’s gate “pH-midpoint position” is undefined.

WHY: With buffering, surface pH may be broadened or nonmonotone. A vague midpoint can be handpicked.

WHAT TO DO: Define the metric now: surface location/average, interpolation rule, pH threshold or midpoint formula, monotonicity requirement, and what happens if there are multiple crossings.

6. WHAT: The raw ring-data gate for the bump disappeared from the updated structure.

WHY: Rebinning SVG vertices does not prove the bump is experimental. C′ and D′ still target R2, so they still depend on whether the bump exists in the original data.

WHAT TO DO: Make raw ring xlsx or documented original-data confirmation an explicit gate before any bump-specific chemistry claim. Otherwise label C′/D′ as digitized-curve sensitivity only.

7. WHAT: Re 9’s tail penalty uses zero as the censoring threshold.

WHY: “Thresholded zero” means below detection/baseline threshold, not mathematically zero. Penalizing all cathodic current below zero can overconstrain the anodic tail and shift the cliff.

WHAT TO DO: Use the actual detection/baseline threshold as the one-sided limit, and run sensitivity over that threshold and `s`.

8. WHAT: Re 11 still describes R3 scoring as a stoichiometric reaction sum, not the measured escape flux.

WHY: The ring observes H2O2 escaping the disk region. With consumption, numerical residuals, and boundary conditions, integrated net reaction should be checked against actual outer-boundary flux, not assumed identical.

WHAT TO DO: Score the outer-boundary H2O2 flux. Report production, consumption, and mass-balance mismatch separately.

9. WHAT: Re 10’s PDE-null criterion is still vague: constrained H3 cannot “beat” an unconstrained same-machinery null on the same fitted objective.

WHY: Without a predefined validation split/stability rule, this becomes interpretive after the fact.

WHAT TO DO: Define the comparison before fitting: free-null Vθ stability under bootstrap/rebinning, constrained-vs-free objective penalty, and out-of-window residual change on held-out or bump-omitted fits.

10. WHAT: C′ says acid re-entry requires the c_H-power factor to be “within a stated range,” but the range is not stated.

WHY: That numeric rule is the actual guard against acid Tafel extrapolation. Leaving it open reintroduces the artifact through judgment calls.

WHAT TO DO: Set the allowed pH or log concentration-factor range before C′ starts, separately for acid-2e and acid-4e.

VERDICT: ISSUES_REMAIN