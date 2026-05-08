1. WHAT: The multi-steric closed-form is a good correction, but your outer-φ formula silently sets `a_O2 = a_H2O2 = 0`.
WHY: The current `THREE_SPECIES_LOGC_BOLTZMANN` has `A_DEFAULT` for O2, H2O2, and H+. If the code uses actual `a_vals`, then `A_dyn_outer` is not just `a_H·H_o`.
WHAT TO DO: Either explicitly set neutral dynamic steric sizes to zero for this fast realignment, or compute `A_dyn_outer = Σ_i a_i c_i_outer` from the actual species config.

2. WHAT: The “byte-equivalent legacy path when `len(counterions)==1 AND a_H==0`” condition will not preserve current production behavior.
WHY: In the current production species config, `a_H` is not zero. That condition routes existing single-counterion Bikerman runs into the new multi-ion machinery.
WHAT TO DO: Preserve legacy with an explicit branch on `len(counterions)==1` unless the caller opts into `counterion_ctx` multi-ion mode. Do not key byte-equivalence on `a_H == 0`.

3. WHAT: The proposed `effective_debye_length()` uses `Σ z² c_bulk`, which is only the ideal dilute linearization around bulk.
WHY: With steric Bikerman and shifted `φ_o`, the correct screening coefficient is the local charge derivative, roughly `-dρ/dφ | outer`, including the shared denominator derivative.
WHAT TO DO: Compute `λ_eff = sqrt(eps / max(-dρ_dφ_outer, floor))` from the same multi-steric closure, analytically or by finite difference. Use that in Stern and linear-Debye IC.

4. WHAT: Phase 2.4 says O2/H2O2 spatial seeds remain simply linear-in-y.
WHY: With steric active, neutral species still feel `μ_steric = -ln θ`; the existing IC adds `log_gamma` to O2 and H2O2 for that reason.
WHAT TO DO: The multi-ion spatial IC must build `gamma_psi(y)` from the shared-theta closure and seed neutral species as `ln(c_outer_i(y)) + log_gamma_psi(y)` unless you deliberately disable sterics for them.

5. WHAT: Pass D’s fixed `K0_R4E = 1e-15 × K0_R2E` only cancels the 4e BV advantage near the anchor.
WHY: The 4e/2e exponential ratio changes strongly across voltage. A single factor can be balanced at `+0.55 V` and still be wildly 4e-dominated elsewhere.
WHAT TO DO: Make Pass D a short ladder, e.g. `1e-18, 1e-16, 1e-14, 1e-12`, and accept the first factor that gives nonzero 2e and 4e at multiple voltages.

6. WHAT: The final done criterion is ambiguous: “Pass A + B + C + D combined ≥15/25” can be read as 15 successes across 100 attempted solves.
WHY: That would be far weaker than the original page-15 sweep requirement.
WHAT TO DO: State it per pass. Minimum defensible fast criterion: Pass A and Pass D each ≥15/25, Pass B anchor plus warm-walk sanity, Pass C clearly diagnostic only unless it also reaches ≥15/25.

7. WHAT: `counterion_ctx["theta_b"]` risks becoming a duplicated derived value.
WHY: If `_bv_common.py`, `boltzmann.py`, Picard, and IC each compute or store theta independently, they will drift.
WHAT TO DO: Store only raw ion/species parameters in `counterion_ctx`; compute `theta_b`, ion concentrations, charge density, and derivatives through one helper used by residual, Picard, Stern, and IC.

VERDICT: ISSUES_REMAIN