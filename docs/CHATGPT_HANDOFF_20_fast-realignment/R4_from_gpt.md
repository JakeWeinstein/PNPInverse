1. WHAT: Your spatial IC still confuses absolute `φ(y)` with diffuse-layer `ψ(y)`.
WHY: The H+ Boltzmann shift is relative to the outer potential, not absolute potential. Existing code uses `log_H = log(H_outer) - ψ + log_gamma`, not `log(H_outer) - φ`. Using `-φ(y)` double-counts `φ_outer` and breaks the `mu_H` cancellation in `logc_muh`.
WHAT TO DO: Split the IC into `φ_init(y) = φ_outer(y) + ψ(y)`. Use `ψ(y)` for Boltzmann shifts and γ. Use absolute `φ_init(y)` only for the Poisson variable and `mu_H = u_H + z_H φ`.

2. WHAT: You still treat O2/H2O2 as bulk-valued in the “outer region.”
WHY: In the existing matched IC, `O_outer(y)` and `P_outer(y)` interpolate from Picard surface values `O_s/P_s` to bulk. At the OHP-side outer edge, the anchors are `O_s/P_s`, not `O_b/P_b`.
WHAT TO DO: Build `c_dyn_outer(y)` from the Picard profiles for all dynamic species, then solve `φ_outer(y)` from multi-ion electroneutrality using those local dynamic concentrations. Do not hardcode neutrals to bulk.

3. WHAT: The bulk `λ_eff = sqrt(eps / Σz²c_bulk)` approximation is too weak now that Stern dominates.
WHY: You admitted the local screening can differ by up to ~20×. Since Stern split controls η and ψ_D, this can put the IC on the wrong branch.
WHAT TO DO: Implement local `λ_eff` from `-dρ/dφ` using the same multi-steric closure. Finite-difference derivative is acceptable and cheap.

4. WHAT: `multi_ion_enabled=False` with multiple counterions would silently take `counterions[0]` through the legacy path.
WHY: That is a catastrophic footgun: one missing flag turns Cs/SO4 into “whatever ion is first.”
WHAT TO DO: Validate config: if `len(boltzmann_counterions) > 1` and `multi_ion_enabled` is false, raise a hard error.

5. WHAT: The done criterion still contradicts itself on Pass C: it says Pass C has `≥5/25 OK`, then says Pass C never gates done.
WHY: Acceptance criteria must be mechanically checkable.
WHAT TO DO: Remove Pass C from done entirely, or make it an explicit required smoke criterion. Do not state both.

VERDICT: ISSUES_REMAIN