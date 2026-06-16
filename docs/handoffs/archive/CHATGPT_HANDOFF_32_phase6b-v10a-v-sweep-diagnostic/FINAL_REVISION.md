# FINAL_REVISION — Critique session 32

- **Topic**: phase6b-v10a-v-sweep-diagnostic
- **Rounds**: 3 (cap; converged early)
- **Final verdict**: APPROVED (R3)
- **Revised artifact**: `/Users/jakeweinstein/.claude/plans/whimsical-tumbling-hoare.md`
- **Session dir**: `docs/handoffs/CHATGPT_HANDOFF_32_phase6b-v10a-v-sweep-diagnostic/`

## Summary

14 issues raised across 3 rounds; all 14 accepted. The plan landed
with structural changes (column rename, locked-rule split into
estimator-validity + physics-filter stages, dropped FD-mismatch
exclusion) plus precise unit/naming fixes (Levich D_O2, σ-gap
floors, slope-scale floor in derivative units, locked-filter
boolean disambiguation, selection-precedence ordering).

## Addressed (14)

### Round 1 — 5 issues

| # | Issue | Fix landing in plan |
|---|---|---|
| R1.1 | `dR_net/dσ_S` mislabeled — actually `(dR/dlogC_S)/(dσ/dlogC_S)` along the Stern-capacitance manifold | "Sensitivity computation" section: renamed column to `dRnet_dsigma_along_stern_capacitance`; intermediate quantities logged; docstring states the column is a total-derivative-along-C_S, not a partial |
| R1.2 | `C_S` perturbation falsely equated with `r_H_El` calibration knob | "Sensitivity computation" section: C_S framed strictly as Stern-capacitance leverage; `dRnet_dr_H_El` diagnostic deferred to Phase A.2 |
| R1.3 | 50 % FD-vs-perturbation exclusion is a hidden fourth filter beyond the locked rule | Exclusion DROPPED. `path_mismatch_relative` logged informationally only. Numerical-quality filter replaced by one-sided slope agreement on the perturbation column itself (≤ 0.25 primary / ≤ 0.50 fallback) |
| R1.4 | Levich helper hardcoded `D_O2 = 2.18e-9` (water lit); codebase has `D_O2 = 1.9e-9` | "Levich limit" code block: imports `D_O2`, `C_O2` from `_bv_common`; docstring updated; corrected value ≈ 5.50 mA/cm² at l_eff = 16 µm |
| R1.5 | `\|cd\|/I_lim_4e` asymmetric for parallel 2e/4e (pure 2e plateau caps at 0.5, only pure 4e reaches 0.9) | Locked rule preserved literal (per acceptance bundle); `o2_flux_levich_ratio` informational indicator added; `locked_current_filter_passes_but_o2_transport_limited` flag surfaces the asymmetry without amending the rule |

### Round 2 — 6 issues

| # | Issue | Fix landing in plan |
|---|---|---|
| R2.1 | `σ_min` adaptive floor's "no-op if all σ-clamped" silently approves unidentifiable perturbations | Absolute floor `σ_abs_min = 1e-4 C/m²` added; `no_valid_stern_capacitance_sensitivity` fail-stop status emitted when no V clears it |
| R2.2 | Quality filter only floored two-sided gap; one-sided denominator could be tiny | Per-side floor `σ_side_min` added; both `\|σ_+ − σ_0\|` and `\|σ_− − σ_0\|` must clear it |
| R2.3 | `ε_quality` (= 0.05 fractional step) reused as slope-scale floor — dimensionally invalid | `sensitivity_floor` redefined in derivative units: `max(1e-12, 1e-3 · max(\|S_+\|, \|S_−\|))`. Dimensionless ε never reused in slope expressions |
| R2.4 | `sensitivity_quality_primary` described as part of the LOCKED rule — it's a hidden fourth filter | `select_v_kin` split into Stage 1 (estimator validity, NOT locked) + Stage 2 (locked rule applied to records with valid estimators). The locked rule body contains only the three locked filters |
| R2.5 | `o2_flux_levich_ratio` had integral-vs-flux mismatch (boundary integral / per-area flux) | Numerator divided by `electrode_area_nondim` (read from live ctx); denominator pulls `domain_height_hat` from `sp.solver_options['bv_convergence']`, not a module global; `abs()` to keep ratio in [0, 1] |
| R2.6 | `locked_filter_passed` name ambiguous (current-only? all three?) | Split into explicit booleans: `locked_sigma_neg_filter_passed`, `locked_current_filter_passed`, `locked_branch_filter_passed`, `locked_three_filters_passed`; Levich-asymmetry warning fires off `locked_current_filter_passed` specifically |

### Round 3 — 3 nits

| # | Issue | Fix landing in plan |
|---|---|---|
| R3.1 | `dRnet_dlogCs` denominator should be `log(1+ε) − log(1−ε)`, not `2ε` | "Sensitivity computation" table entry: log denominator uses the exact form |
| R3.2 | `σ_side_min` should inherit the adaptive scale, not just absolute floor | Updated to `σ_side_min = 0.5 · max(σ_abs_min, 0.1 · median_nonzero_delta_sigma)` |
| R3.3 | Status precedence: `abort_to_v10c` should be checked BEFORE `no_valid_stern_capacitance_sensitivity` when both could apply | "Selection precedence" section added: order is (1) abort_to_v10c, (2) no_valid_stern_capacitance_sensitivity, (3) locked rule (Stage 2) |

## Defended (0)

None. Every issue raised was accepted as substantively correct.

## Unresolved (0)

None. Loop converged on round 3 (cap) with VERDICT: APPROVED.

## Notes for downstream readers

- The plan's "Critique provenance" section embeds a per-round
  summary so a reader who picks up the plan in 6 months sees the
  decision trail without having to chase this session dir.
- The locked V_kin rule from
  `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` step 4
  was never modified — every fix is upstream of it (estimator
  validity) or supplementary to it (informational indicators).
- The most consequential structural fix is **R1.3 + R2.4**: dropping
  the 50 % FD-mismatch exclusion and splitting estimator quality
  from the locked physics rule. The pre-critique plan would have
  silently rejected V_kin candidates based on path-dependence
  artifacts and called it part of the locked filter set.
