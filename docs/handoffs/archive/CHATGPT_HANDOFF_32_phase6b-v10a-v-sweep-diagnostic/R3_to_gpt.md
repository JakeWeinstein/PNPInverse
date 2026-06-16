# Round 3 counterreply — V-sweep diagnostic plan

All six points accepted; the fixes are mechanical and self-contained.
Counterreplies + revised plan section below.

## 1. Acknowledgment per issue

### Re your point 1 — σ_min adaptive floor can silently approve unidentifiable perturbation. **Accept.**

The "no-op if all V are σ-clamped" escape hatch was exactly the wrong
move; if `C_S` doesn't move σ_S anywhere, the Stern-capacitance
diagnostic is unidentifiable and the right action is failure, not
silent pass.

**Fix:**

- Add absolute floor `σ_abs_min` in physical C/m². Concrete value:
  `σ_abs_min = 1e-4 C/m²` (≈ 100 µC/m²; ~0.5 % of typical Stern
  surface charge magnitude at V_RHE ≈ −0.3 V on this stack, and
  well above Newton residual noise floor at SNES rtol=1e-10).
- Each V must satisfy `|σ_+ − σ_−| ≥ max(σ_abs_min, 0.1 × median_nonzero_delta_sigma)`.
- If *no* V clears the absolute floor, driver emits
  `no_valid_stern_capacitance_sensitivity` status (parallel to
  `abort_to_v10c`) — V_kin = None, the JSON records the failure,
  and downstream callers know to retry with larger ε or a different
  perturbation knob.

### Re your point 2 — check each one-sided denominator. **Accept.**

The two-sided floor only protects the *central* score; one side
collapsing to ≈ σ_0 corrupts `S_plus` or `S_minus` independently of
the two-sided gap.

**Fix:**

- Per-side floor: `|σ_+ − σ_0| ≥ σ_side_min` AND `|σ_− − σ_0| ≥ σ_side_min`.
- `σ_side_min = σ_abs_min / 2` (half the two-sided floor — each side
  contributes ~half the spread in the symmetric case).
- Two-sided floor on `|σ_+ − σ_−|` retained for the central
  difference score itself.
- Records `denominator_one_sided_minus = σ_− − σ_0`,
  `denominator_one_sided_plus = σ_+ − σ_0`, and
  `denominator_two_sided = σ_+ − σ_−` so the reader can audit each.

### Re your point 3 — ε_quality and path_mismatch_relative have wrong units. **Accept.**

Reusing the perturbation fraction (0.05) as the slope-scale floor was
dimensionally invalid: the denominator of `|FD − perturb| / max(|FD|, |perturb|, ε)`
has units of `R_net / σ_S` (i.e. derivative units), not a fraction.

**Fix:**

- Drop `ε_quality` and `ε` (the dimensionless 0.05) from any
  expression with derivative dimensions.
- Define `sensitivity_floor` in derivative units explicitly. Two
  options, both implemented; driver uses whichever is larger
  per-V:
    - `sensitivity_floor_abs = 1e-12` in nondim-rate units per
      (C/m²) — a numerical-noise scale far below any physically
      meaningful sensitivity.
    - `sensitivity_floor_adaptive = max(|S_+|, |S_−|) × 1e-3` —
      tracks the local scale so a V with a genuinely huge slope
      isn't flagged just because of FD's coarser response.
- `path_mismatch_relative = |FD − perturb| / max(|FD|, |perturb|, sensitivity_floor)`
  with `sensitivity_floor = max(sensitivity_floor_abs, sensitivity_floor_adaptive)`.
- `one_sided_disagreement = |S_plus − S_minus| / max(|S_plus|, |S_minus|, sensitivity_floor)`
  with the same floor.

### Re your point 4 — quality filter is not part of the locked rule. **Accept.**

The R2 docstring conflated numerical-quality gating with the locked
physics rule. Need to split clearly: quality validates the
estimator; the locked rule applies to records with valid estimators.

**Fix to `select_v_kin` shape:**

```python
def select_v_kin(per_v_records, *, i_lim_4e_mA_cm2):
    """
    Two-stage selection.

    Stage 1 — estimator validity (per-V quality gate, NOT in the
              locked rule):
        valid_estimator(V) iff:
          - perturbation_converged(V)
          - |σ_+ − σ_−| ≥ max(σ_abs_min, 0.1 · median_nonzero)
          - |σ_+ − σ_0| ≥ σ_side_min
          - |σ_− − σ_0| ≥ σ_side_min
          - one_sided_disagreement ≤ 0.25 (primary) or ≤ 0.50 (fallback)

    If no V has valid_estimator(V), return
    VKinDecision(v_kin=None, no_valid_stern_capacitance_sensitivity=True, ...)

    Stage 2 — apply the LOCKED rule literally to V with valid_estimator(V):
        argmax(|dRnet_dsigma_along_stern_capacitance|) subject to {
            σ_S(V) < 0,
            |cd(V)| / I_lim_4e < 0.9,
            R_2e(V)/(R_2e(V) + R_4e(V)) ∈ [0.05, 0.95],
        }
        with fallback (drop branch filter; ALSO drop primary→fallback
        on the Stage 1 quality threshold) and abort_to_v10c when no V
        has σ_S < 0.
    """
```

The locked rule body in stage 2 contains *only* the three locked
filters. The estimator-quality gate is upstream (stage 1) and labeled
as such.

### Re your point 5 — o2_flux_levich_ratio has integral-vs-flux mismatch. **Accept.**

Boundary integral has units `nondim_rate · nondim_area`; the Levich
flux is `nondim_rate / nondim_area`. Need to normalise.

**Fix:**

```python
def _compute_o2_flux_levich_ratio(record, *, electrode_area_nondim,
                                   domain_height_hat):
    """O₂ transport-limit indicator, branch-selectivity-independent.

    Numerator: average boundary O₂ consumption rate, per unit area
        (R_2e_current_nondim + R_4e_current_nondim) / electrode_area_nondim
        — both R_j_current_nondim values are boundary integrals from
        collect_v10a_rung_diagnostics, dividing by area gives the
        per-area mean.

    Denominator: bulk Levich O₂ flux (per area)
        D_O2_HAT · C_O2_HAT / domain_height_hat
        — the supply flux from the bulk boundary across the diffusion
        layer.  D_O2_HAT = 1, C_O2_HAT = 1 in the current nondim, so
        this reduces to 1 / domain_height_hat.

    Ratio is in [0, 1] at the transport limit regardless of R_2e/R_4e
    split.  `electrode_area_nondim` is read from the actual ctx via
    `float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))` at
    the time the records are collected.  `domain_height_hat` is read
    from `sp.solver_options['bv_convergence']['domain_height_hat']`,
    not a module global.
    """
    from scripts._bv_common import D_O2_HAT, C_O2_HAT  # nondim, currently 1.0
    o2_consumption_nondim = (
        record["R_2e_current_nondim"] + record["R_4e_current_nondim"]
    )
    o2_consumption_per_area = o2_consumption_nondim / electrode_area_nondim
    levich_flux = D_O2_HAT * C_O2_HAT / domain_height_hat
    # Sign: R_*_current_nondim are positive for cathodic consumption
    # (boundary integral of the BV rate expression which is positive
    # in the reduction direction).  Use abs to keep the ratio in
    # [0, 1] regardless of which sign convention the diagnostics
    # collector emits.
    return abs(o2_consumption_per_area) / levich_flux
```

Plan now requires the driver to pass `electrode_area_nondim` and
`domain_height_hat` into the V_kin selection / O₂-ratio block;
the driver reads both off the live ctx at extraction time.

### Re your point 6 — name locked_filter_passed precisely. **Accept.**

**Fix:**

```python
# Per-V booleans (explicit, no ambiguous shorthand):
record["locked_current_filter_passed"] = (
    abs(cd_mA_cm2) / i_lim_4e_mA_cm2 < 0.9
)
record["locked_sigma_neg_filter_passed"] = (sigma_S_C_per_m2 < 0.0)
record["locked_branch_filter_passed"] = (
    0.05 <= r2e / max(r2e + r4e, 1e-30) <= 0.95
)
record["locked_three_filters_passed"] = (
    record["locked_current_filter_passed"]
    and record["locked_sigma_neg_filter_passed"]
    and record["locked_branch_filter_passed"]
)

# Levich-asymmetry warning fires off the CURRENT filter only:
record["locked_current_filter_passes_but_o2_transport_limited"] = (
    record["locked_current_filter_passed"]
    and record["o2_flux_levich_ratio"] > 0.9
)
```

## 2. Updated artifact

Changes since R2:

* Added absolute floor `σ_abs_min = 1e-4 C/m²` + fail-stop
  `no_valid_stern_capacitance_sensitivity` when no V clears it.
* Per-side floor `σ_side_min = σ_abs_min / 2` on each one-sided
  denominator, in addition to the two-sided floor on the central
  difference.
* `sensitivity_floor` defined in derivative units (max of an
  absolute `1e-12` and an adaptive `1e-3 × max(|S_+|, |S_−|)`); no
  dimensionless fractions in slope expressions.
* `select_v_kin` split into Stage 1 (estimator validity, NOT locked)
  + Stage 2 (locked physics rule applied to valid-estimator
  records).  Locked rule body has only the three locked filters.
* `o2_flux_levich_ratio` normalised by `electrode_area_nondim`;
  reads `domain_height_hat` from live config, not a module global;
  uses `abs()` to guarantee the [0, 1] range regardless of sign
  convention.
* Per-V locked-filter booleans split into
  `locked_current_filter_passed`, `locked_sigma_neg_filter_passed`,
  `locked_branch_filter_passed`, `locked_three_filters_passed`,
  with the Levich-asymmetry warning explicitly tied to
  `locked_current_filter_passed`.

### Revised numerical-quality + selection block

```markdown
## Estimator validity (Stage 1, NOT part of the locked rule)

For each V, the perturbation estimator
`dRnet_dsigma_along_stern_capacitance` is valid iff ALL hold:

1. `perturbation_converged`: both `C_S·(1±ε)` solves converged.
2. Two-sided gap: `|σ_+ − σ_−| ≥ max(σ_abs_min, 0.1 · median_nonzero)`
   where `σ_abs_min = 1e-4 C/m²`.
3. Per-side gap: `|σ_+ − σ_0| ≥ σ_side_min` AND
   `|σ_− − σ_0| ≥ σ_side_min` with `σ_side_min = σ_abs_min / 2`.
4. One-sided slope agreement:
   `|S_+ − S_−| / max(|S_+|, |S_−|, sensitivity_floor) ≤ 0.25`
   (primary) or `≤ 0.50` (fallback). `sensitivity_floor` is in
   derivative units:
       `sensitivity_floor = max(1e-12, 1e-3 · max(|S_+|, |S_−|))`.

If no V has a valid estimator at *any* quality tier, the driver emits
`no_valid_stern_capacitance_sensitivity` and `v_kin = None`. JSON
records all per-V flags so the reader can audit.

`path_mismatch_relative = |FD − perturb| / max(|FD|, |perturb|, sensitivity_floor)`
is logged informationally only — never used as a filter.

## Locked V_kin rule (Stage 2)

Apply to V with valid estimators:

```python
def select_v_kin(per_v_records, *, i_lim_4e_mA_cm2,
                  electrode_area_nondim, domain_height_hat):
    # Stage 1 — estimator validity (out of locked rule)
    primary_valid = [
        r for r in per_v_records
        if r["perturbation_converged"]
           and r["two_sided_gap_ok"]
           and r["per_side_gap_ok"]
           and r["one_sided_disagreement"] <= 0.25
    ]
    fallback_valid = [
        r for r in per_v_records
        if r["perturbation_converged"]
           and r["two_sided_gap_ok"]
           and r["per_side_gap_ok"]
           and r["one_sided_disagreement"] <= 0.50
    ]
    if not fallback_valid:
        return VKinDecision(
            v_kin=None,
            no_valid_stern_capacitance_sensitivity=True,
            ...
        )

    # Stage 2 — LOCKED rule on the valid-estimator records
    #   argmax(|dRnet_dsigma_along_stern_capacitance|)
    #   subject to { σ_S < 0, |cd|/I_lim_4e < 0.9, branch ∈ [0.05, 0.95] }
    def passes_locked(r):
        return (
            r["locked_sigma_neg_filter_passed"]
            and r["locked_current_filter_passed"]
            and r["locked_branch_filter_passed"]
        )
    primary_candidates = [r for r in primary_valid if passes_locked(r)]
    if primary_candidates:
        best = max(
            primary_candidates,
            key=lambda r: abs(r["dRnet_dsigma_along_stern_capacitance"]),
        )
        return VKinDecision(
            v_kin=best["v_rhe"],
            score=abs(best["dRnet_dsigma_along_stern_capacitance"]),
            fallback_used=False,
            ...
        )

    # Fallback: drop branch filter; relax to fallback quality tier
    fallback_candidates = [
        r for r in fallback_valid
        if r["locked_sigma_neg_filter_passed"]
           and r["locked_current_filter_passed"]
    ]
    if fallback_candidates:
        best = max(
            fallback_candidates,
            key=lambda r: abs(r["dRnet_dsigma_along_stern_capacitance"]),
        )
        return VKinDecision(
            v_kin=best["v_rhe"],
            score=abs(best["dRnet_dsigma_along_stern_capacitance"]),
            fallback_used=True,
            ...
        )

    # Fail-stop: no V has σ_S < 0
    if not any(r["locked_sigma_neg_filter_passed"] for r in per_v_records):
        return VKinDecision(v_kin=None, abort_to_v10c=True, ...)

    # No candidate cleared the locked rule even in fallback
    return VKinDecision(v_kin=None, no_candidate_passed_locked_rule=True, ...)
```
```

## 3. Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Scope reminder: still narrow on the three subtle physics/numerics
points (perturbation chain rule, FD/perturb cross-check, Levich
formula + parallel-2e/4e asymmetry). Don't critique the locked V_kin
rule itself, driver structure, or V_RHE grid resolution.

This is the third (final) round under the configured cap; if material
blockers remain, please be explicit so the auto-revise step can
record them as Unresolved.
