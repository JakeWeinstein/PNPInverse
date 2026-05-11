The conceptual fixes mostly land. I would not re-issue points 1, 2, or the `D_O2` typo. Remaining problems are in the new numerical-quality machinery and the O₂-flux diagnostic.

1. **Point 3: the adaptive `σ_min` floor can silently approve an unidentifiable perturbation.**

WHAT: `σ_min = median(|σ_+ − σ_-|)/10`, with “if all V are σ_S-clamped, this filter is no-op,” is wrong.

WHY: If `C_S` barely moves `σ_S` anywhere, the Stern-capacitance-manifold derivative is not identifiable. Making the floor vanish turns division by solver noise into a candidate score.

WHAT: Add an absolute floor and fail/flag when the whole sweep is σ-insensitive. Example: require `|σ_+ − σ_-| >= max(σ_abs_min, 0.1 * median_nonzero_delta_sigma)`, with `σ_abs_min` in physical C/m². If no V clears it, emit `no_valid_stern_capacitance_sensitivity`, not a normal V_kin.

2. **Point 3: check each one-sided denominator, not just `|σ_+ − σ_-|`.**

WHAT: The quality filter only floors the two-sided separation. One side can still have `σ_+ ≈ σ_0` or `σ_- ≈ σ_0`, making `S_plus` or `S_minus` garbage.

WHY: Then `one_sided_disagreement` is itself contaminated by a bad denominator.

WHAT: Require both `|σ_+ − σ_0| >= σ_side_min` and `|σ_- − σ_0| >= σ_side_min`. Use the two-sided floor only for the central score.

3. **Point 3: `ε_quality` and `path_mismatch_relative` need unit-correct floors.**

WHAT: `max(|FD|, |perturb|, ε)` is dimensionally invalid if `ε` is the 0.05 fractional C_S step. Same issue for `ε_quality` unless it is explicitly a slope-scale floor.

WHY: A unitless number cannot regularize a derivative with units of `R_net / (C/m²)`.

WHAT: Define separate slope floors, e.g. `sensitivity_floor = 1e-12` in the actual derivative units or an adaptive floor from the nonzero slope distribution. Do not reuse the perturbation fraction.

4. **Point 3: don’t describe `sensitivity_quality_primary` as part of the locked rule.**

WHAT: The revised docstring says it “implements the LOCKED rule literally” but then includes `sensitivity_quality_primary` inside the subject-to set.

WHY: That is still a fourth filter. It is more defensible than FD mismatch, but it is not literally in the locked rule.

WHAT: Split the wording: first compute a valid sensitivity estimator; then apply the locked rule to records with a valid score. If no valid-score records exist, report estimator failure separately. Do not present quality gating as a locked physics filter.

5. **Point 5: `o2_flux_levich_ratio` has an integral-vs-flux mismatch.**

WHAT: `R_2e_current_nondim + R_4e_current_nondim` from diagnostics is a boundary integral. `D_O2_HAT * C_O2_HAT / domain_height_hat` is a per-area nondim flux.

WHY: Unless the electrode measure is exactly 1, the ratio is dimensionally wrong and the transport-limited flag can be false.

WHAT: Either divide the numerator by electrode area or multiply the denominator by electrode area. Prefer:

`o2_flux_levich_ratio = ((R2_int + R4_int) / electrode_area) / (D_O2_HAT * C_O2_HAT / domain_height_hat)`

Also pull `domain_height_hat` from the actual run config, not a module global, and handle sign with positive O₂ consumption.

6. **Point 5: name `locked_filter_passed` precisely.**

WHAT: The flag `locked_filter_passes_but_o2_transport_limited` depends on `locked_filter_passed`, but that name is ambiguous: current filter only, all three locked filters, or selected-candidate pass?

WHY: If it means all filters, you undercount the Levich asymmetry. If it means only current filter, the name is misleading.

WHAT: Use explicit booleans: `locked_current_filter_passed`, `locked_three_filters_passed`, and then define the warning off the one you actually mean.

VERDICT: ISSUES_REMAIN