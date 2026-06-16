# Round 4 counterreply — v10a' next-steps (cap raised to 5)

The round cap was raised from 3 to 5 because the user wants the
plan reviewed all the way to **VERDICT: APPROVED** rather than
auto-revised post-cap.  All 5 of your R3 issues have been
incorporated into the plan via auto-revise; this round shows you
the revised plan sections so you can verify the fixes landed
correctly and verdict APPROVED (or call out any that didn't).

## 1. Acknowledgment per issue

### Re point 1 — `amplification_from_c_K` expectation. **Accepted.**

You're right: C_K_HAT_bulk ≈ 166.6 (= 199.9/1.2 from
`scripts/_bv_common.py:757`), and v10a's
`F0/k_hyd ∈ {52, 67, 101, 129, 226, 414, 732}` across V_RHE
= +0.55 → −0.50.  With `<10^(−ΔpKa)> ≈ 1` (since pka_shift_avg
is ~1e-6 to 1e-5), the implied `<c_K>` ranges from ~52 (V=+0.55,
3.2× depletion from bulk) to ~732 (V=−0.50, 4.4× enrichment).

So `amplification_from_c_K` runs **0.31 (V=+0.55) → 4.39 (V=−0.50)**
— K+ enrichment is already load-bearing in v10a, **not** a new
v10a' effect from C_S=0.20.

**Plan now says** (Sanity checks section, point 4):

> "K+ enrichment is **already load-bearing in v10a** — not a new
> effect from C_S=0.20."
> Expected pattern:
> - `amplification_from_singh ≈ 1` (Singh ΔpKa is tiny; ~1e-6 to 1e-5).
> - `amplification_from_c_K`: ~0.31 at V=+0.55 (K+ depletion at anode),
>   rising through ~1 near V=0, to ~4.39 at V=−0.50 (4× bulk enrichment).

This is now documented as the expected baseline for the v10a' run.

### Re point 2 — R_4e log decomposition not solver-faithful. **Accepted.**

You're right that:
- With Stern enabled, `η_raw = phi_applied − phi_boundary − E_eq`,
  not `V_RHE − E_eq`.
- Clip is on `eta_scaled = η/V_T` BEFORE the α·n_e multiplication
  (`forms_logc.py:_build_eta_clipped`).
- Raw 4e exponent magnitude does NOT itself indicate clipping is
  active.

**Plan now emits** (Enhanced diagnostic emissions section):

```python
diag["R_4e_decomposition_log"] = {
    "log_k0":                 ln(k0_R4e),
    "eta_scaled_raw_avg":     <η/V_T> over electrode boundary,
    "eta_scaled_min":         min(η/V_T) over boundary,
    "eta_scaled_max":         max(η/V_T) over boundary,
    "eta_scaled_clipped_avg": <clip(η/V_T, -100, 100)>,
    "exponent_clip_active":   any boundary point has |η/V_T| > 100,
    "log_bv_clipped_avg":     -α·n_e · <eta_scaled_clipped>,
    "n_e_log_c_H_factor_avg": n_e · <ln(c_H/c_H_ref)>,
    "log_R4e_predicted":      sum of the above (scalar approximation),
    "log_R4e_measured":       ln(R_4e_current_nondim) if R_4e > 0 else None,
}
```

With an inline label in the plan:

> "This decomposition is a scalar approximation — label as such;
> the `log_R4e_measured` field is authoritative."

`η_raw` correctly defined as `phi_applied − phi_boundary − E_eq`
in the plan's comment.

### Re point 3 — Case F's φ-perturbation fallback. **Accepted.**

You're right: switching to φ_applied perturbation measures
`dRnet/dφ`, not `dRnet/dσ` along the Stern-cap manifold.  Different
derivative; cannot validate the same V_kin score.

**Plan now says** (Decision tree Case F):

> "Per R3 #3, do **NOT** switch to a φ-perturbation knob — that
> measures a different derivative. Instead: increase the C_S
> perturbation ε from 0.05 to 0.10 (or 0.15) and rerun. If still
> `no_valid_stern_capacitance_sensitivity`, escalate as
> 'C_S manifold unidentifiable at this regime'."

φ-perturbation is **not** used in V_kin selection; reserved as
auxiliary diagnostic only.

### Re point 4 — Case G's invalid inference. **Accepted.**

You're right: `no_candidate_passed_locked_rule + no artifact flag`
does NOT imply "σ<0 ∧ o2lev<0.9 holds somewhere".  The artifact
flag only fires when `current_passed=True AND o2lev>0.9`.  Absence
of the flag could mean current failed everywhere (artifact flag
trivially false because current_passed is False).

**Plan now says** (Decision tree Case G):

> "Per R3 #4, do **NOT** infer 'σ<0 + o2lev<0.9 holds somewhere'.
> Instead: read the per-filter failure matrix from
> `per_v_decisions`:
> - counts of `locked_sigma_neg_filter_passed = True`
> - counts of `locked_current_filter_passed = True`
> - counts of `locked_branch_filter_passed = True`
> - counts of `o2_flux_levich_ratio > 0.9`
> - counts of `primary_valid` / `fallback_valid`
> Route based on the actual blocking filter:
> - If `sigma_neg` count = 0 → would have been Case E (abort).
> - If `current_passed` count = 0 → bracket sweep (Case B path).
> - If `branch_passed` count = 0 → bracket sweep (Case B path);
>   bias bracket point selection to where x_4e is closest to
>   [0.05, 0.95] in the v10a record.
> - If `*_valid` count = 0 → Case F.
> - Multiple zero counts → bracket sweep + escalate if no factor
>   opens any single failing filter."

### Re point 5 — Routing precedence ambiguous. **Accepted.**

You're right: a returned V_kin with `cap_dominated=True` could
satisfy both "Case A: clean V_kin → Phase A.2" and "v10b routing
threshold met → v10b".  Auto-revise step can't decide.

**Plan now has** an explicit Step 0 precedence section:

> "After `select_v_kin()` returns a `v_kin`, first compute on the
> selected V's record:
>   - `is_transport_artifact = locked_current_filter_passes_but_o2_transport_limited`
>   - `is_cap_dominated_routing = (
>         all σ<0 fallback-valid V satisfy
>         denom_cap/denom_total > 0.8 AND θ > 0.9 AND |sensS| < 0.10
>     )`
>
> If `is_transport_artifact`: route to **Case B** below regardless
> of what the locked rule said about 'clean V_kin'.
>
> Else if `is_cap_dominated_routing`: route to **v10b** (literature
> calibration) NOT to Phase A.2 — even if `select_v_kin` returned
> a nominally 'clean' V_kin.
>
> Else if neither: proceed to Case A."

Order: artifact check → cap-dominance check → Phase A.2.
Deterministic.

## 2. Updated artifact (changes since R3)

All 5 R3 issues incorporated above are in the live plan file at
`/Users/jakeweinstein/.claude/plans/sparkly-gilded-pasteur.md`.
Specifically:

* **Sanity checks section** now states "K+ enrichment is already
  load-bearing in v10a — not a new effect from C_S=0.20" with
  the 0.31 → 4.39 expected pattern (R3.1).
* **R_4e log decomposition** is labeled as scalar approximation
  with authoritative `log_R4e_measured`; emits boundary
  `eta_scaled` avg/min/max + `exponent_clip_active` (R3.2).
* **Case F** restricts the fallback to "increase ε" — no knob
  switch (R3.3).
* **Case G** uses per-filter failure matrix inspection — no
  invalid inferences (R3.4).
* **Step 0 precedence guards** added before Cases A–G; order
  pinned (R3.5).

In addition to the R3 fixes, the plan has the cumulative R1+R2
revisions:

* Single-run K0_R4e_factor = `1e-14` (was 1e-18; per R1.1 this is
  the V=−0.10 branch-pass probe, not a plateau-shift target).
* Bracket sweep `{1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24}`
  (was the sparse 5-point bracket missing V=−0.10 window; per R1.8).
* Decision tree expanded to 7 cases (A–G; per R2.2).
* "BV-controlled" softened to "passes O₂ sanity check" (R2.3).
* F₀ decomposition uses boundary averages of products
  (Jensen-safe; R2.5).
* `denominator_cap dominates` threshold pinned at
  `>0.8 ∧ θ>0.9 ∧ |sensS|<0.10` (R2.6).
* Attribution disclaimer added (combined C_S + K0 change
  intentional; routing works regardless; R2.7).

## 3. Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed.

Round 4 of 5.  We're aiming for **VERDICT: APPROVED** — if
remaining concerns are minor nitpicks (not blocking), please verdict
APPROVED and call them out for the record.  If genuine blockers
remain, verdict ISSUES_REMAIN and we run round 5.

Same numbered format + verdict line:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Scope reminder: still narrow on the three subtle physics/parameter-
coupling points. Don't critique driver structure, test list, or
file modifications.
