# Round 3 counterreply — v10a' next-steps

All 7 points accepted; no defenses this round. Mechanical fixes
applied.

## 1. Acknowledgment per issue

### Re point 1 — "`1e-14` sits in two windows" was wrong. **Accept.**

V=−0.30 window is [1e-18, 3e-16]; 1e-14 is 30× above its upper edge.
1e-14 is a V=−0.10-window probe only.

**Fix:** Rewrite the framing — "1e-14 targets the V=−0.10 mixed
window only (the V with highest |sensS| in v10a). The dense
bracket sweep is the path to V=−0.30 and V=−0.50 coverage."

### Re point 2 — `o2lev > 0.9` artifact branch needs a deterministic next action. **Accept.**

**Fix:** Replace the "manual review" wording with:

```
After v10a' single run (factor = 1e-14, C_S = 0.20):

  Case A — clean V_kin returned, locked_current_filter_passes_but_o2_transport_limited=False:
    → Phase A.2 at V_kin.  Done.

  Case B — V_kin returned BUT locked_current_filter_passes_but_o2_transport_limited=True (artifact):
    → "current-filter artifact, transport-limited at returned V".
    → Launch the dense bracket sweep
      {1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24}.

  Case C — bracket sweep returns at least one clean V_kin (no artifact):
    → Pick the bracket point whose V_kin has highest |sensS| AND
      the locked filters pass without o2-transport artifact.
    → Phase A.2 at that V_kin.

  Case D — every bracket point either returns no_candidate_passed_locked_rule
           OR returns an artifact candidate:
    → escalate to acceptance bundle review.  Document as
      "locked current filter satisfied only by transport-limited
      mixed-branch plateaus across {1e-10 ... 1e-24}; locked rule
      unsatisfiable for this stack at C_S = 0.20".

  Case E — decision returns abort_to_v10c (no V has σ_S < 0):
    → C_S = 0.20 didn't open the cathodic σ_S region.
    → Launch v10c proper: C_S bracket sweep {0.10, 0.20, 0.30, 0.50} F/m².

  Case F — decision returns no_valid_stern_capacitance_sensitivity:
    → Perturbation estimator collapsed → C_S manifold is not
      identifiable at this parameter regime.
    → Increase perturbation ε from 0.05 to 0.10, rerun.  If still
      no_valid → switch perturbation knob (e.g. perturb φ_applied
      directly, not C_S).

  Case G — decision returns no_candidate_passed_locked_rule but
           NO V has the artifact flag set:
    → at least one V passes σ<0 AND has o2lev<0.9, but no V passes
      the full three-filter set.  Read the branch filter per V; if
      x_4e > 0.95 everywhere, K0_R4e is still too high — go to
      Case B (bracket sweep).  If x_4e < 0.05 everywhere, K0_R4e
      is too low — bracket sweep but bias upward.
```

### Re point 3 — "genuine (BV-controlled)" too strong. **Accept.**

`o2lev ≤ 0.9` is a sanity pass, not proof of BV control. θ + cap
terms must also be checked.

**Fix:** Reword to "passes the O₂-transport sanity check (a
necessary, not sufficient, indicator of BV control)". Add note:
"Interpretation of V_kin remains conditional on the full per-V
record — θ, denominator_cap fraction, decomposition fields."

### Re point 4 — R_4e decomposition raw `exp` would clip-mismatch. **Accept.**

`EXPONENT_CLIP = 100` is on `eta_scaled = η/V_T` BEFORE the α·n_e
multiplication (CLAUDE.md Hard Rule 2; `forms_logc.py:_build_eta_clipped`).
At cathodic V the raw 4e exponent (α·n_e·|η|/V_T = 2·1.33/0.0257 ≈ 103)
exceeds the clip. Also: the BV uses log-rate form (`bv_log_rate=True`),
so the actual code path emits `log(R_j)` directly, then exponentiates;
re-computing in linear space won't match the solver.

**Fix:** Emit log-space fields, mirroring the solver:

```python
diag["R_4e_decomposition_log"] = {
    "log_k0":              float(np.log(k0_R4e)),
    "log_bv_raw":          float(-alpha_R4E * n_e_R4E * eta_R4e / V_T),
    "eta_scaled_raw":      float(eta_R4e / V_T),
    "eta_scaled_clipped":  float(np.clip(eta_R4e / V_T, -100.0, 100.0)),
    "exponent_clip_active": bool(abs(eta_R4e / V_T) > 100.0),
    "log_bv_clipped":      float(-alpha_R4E * n_e_R4E *
                                  np.clip(eta_R4e/V_T, -100.0, 100.0)),
    "n_e_log_c_H_factor":  float(n_e_R4E * np.log(c_H_avg / c_H_ref)),
    "log_R4e_predicted":   sum of the above logs,
    "log_R4e_measured":    float(np.log(R_4e_current_nondim))
                            if R_4e_current_nondim > 0 else None,
}
```

Reader can confirm `log_R4e_predicted ≈ log_R4e_measured` (within
clip-adjusted tolerance) and decompose the suppression sources by
inspecting the additive log terms.

### Re point 5 — F0 decomposition averaging error (Jensen). **Accept.**

`10^(−<ΔpKa>)` ≠ `<10^(−ΔpKa)>` in general. The two values agree
at the order of magnitude only when `Var(ΔpKa)` is small. Today
they agree because `ΔpKa` is uniformly tiny, but in v10b that breaks.

**Fix:** Emit boundary averages of the actual factors and their
products, not factor-of-averages:

```python
diag["F0_decomposition"] = {
    # Boundary averages of the actual factors (Jensen-safe):
    "c_K_avg":                   <c_K> over electrode boundary,
    "pka_factor_avg":            <10^(-ΔpKa)>,
    "c_K_pka_product_avg":       <c_K · 10^(-ΔpKa)>,
    "F0_total":                  k_hyd * c_K_pka_product_avg,

    # Counterfactual approximations (label clearly):
    "F0_counterfactual_c_K_bulk_only":
        k_hyd * c_K_bulk * pka_factor_avg,  # what F0 would be if c_K stayed at bulk
    "F0_counterfactual_no_singh":
        k_hyd * c_K_avg,                     # what F0 would be if pka_factor = 1

    "amplification_from_c_K":
        F0_total / (k_hyd * c_K_bulk * pka_factor_avg),  # by how much K+ enrichment amplifies
    "amplification_from_singh":
        F0_total / (k_hyd * c_K_avg),                     # by how much Singh ΔpKa amplifies
}
```

The two `amplification_from_*` ratios are the actual answers to
"which mechanism dominates?". Today both should be ~1 (no Singh
ΔpKa effect, no significant K+ depletion). In v10b with C_S = 0.20
the K+ amplification may grow.

### Re point 6 — "denominator_cap dominates" undefined. **Accept.**

**Fix:** Pin the threshold now:

```
v10b-prerequisite-signal (cap dominates):
  denominator_cap / denominator_total > 0.8
  AND theta > 0.9
  AND |dRnet_dsigma_along_stern_capacitance| < 0.10
  (the sensitivity threshold is below the v10a |sensS| baseline of
   0.21 at V=-0.10, so "cap kills σ_S leverage" is concrete)

  ALL THREE conditions met at every σ_S<0, fallback-tier-valid V
  in the v10a' record → v10b literature calibration is the load-
  bearing next step, NOT v10c bracket sweep on C_S.

ELSE: v10c (C_S bracket sweep) is the load-bearing next step.
```

This makes the v10b-vs-v10c routing deterministic.

### Re point 7 — combined C_S + K0 change loses attribution. **Accept.**

Genuinely fair point. If θ worsens in the v10a' run, we can't say
whether C_S = 0.20 or K0 = 1e-14 is the cause. v10a' is operational
(get a V_kin), not attribution-clean.

**Fix:** Two options, plan picks Option (b):

(a) **Add a C_S=0.20, factor=1 control run.** Wall budget: +1.5 h
(another anchor + walk at the new C_S). Lets us isolate the C_S
effect. **Skipped** because it just reruns the v10a scenario at a
slightly different C_S — the answer is already largely visible by
re-reading v10a's σ_S/θ profile and extrapolating.

(b) **Explicitly disclaim attribution and route around it.**
v10a' is operational. If the diagnostic finds a clean V_kin, we
don't *need* attribution. If it returns the v10b-prerequisite
signal (cap dominates at all σ<0 V), the v10b calibration anchors
Γ_max / k_des regardless of which knob caused the saturation. The
attribution conversation only matters if we want to *roll back*
C_S = 0.20 — which we shouldn't, since 0.20 is the literature-
anchored production target.

**Fix in plan:** Add an explicit disclaimer:

```
## Attribution disclaimer

v10a' changes BOTH C_S (0.10 → 0.20) AND adds the K0_R4e factor
knob.  If the run's outcome is operational success (clean V_kin),
attribution isn't needed.  If the outcome is the v10b-prerequisite
signal (cap dominates), we route to v10b regardless of which knob
caused it.  The only scenario where attribution matters is if we
were considering rolling back C_S = 0.20 — we are not, since 0.20
is the new literature-anchored production target per
.research/cmk3-stern-capacitance/SUMMARY.md.

If a future C_S audit becomes necessary, the C_S=0.20+factor=1
control run can be added as a follow-up at any time (wall ~1.5 h).
```

## 2. Updated artifact (changes since R2)

All 7 fixes applied. Summary of plan deltas this round:

* **Decision tree expanded to seven explicit branches A–G** (was
  three) so every diagnostic outcome maps to a deterministic next
  action.
* **"BV-controlled" → "passes O₂-transport sanity check"**;
  interpretation conditional on θ + cap fraction + decompositions.
* **R_4e decomposition emitted in log-space** with clip-awareness;
  matches the solver's bv_log_rate path.
* **F₀ decomposition uses boundary averages of products** (Jensen-
  safe); explicit `amplification_from_c_K` vs
  `amplification_from_singh` ratios.
* **`denominator_cap dominates` threshold pinned**: `≥0.8 AND
  θ≥0.9 AND |sensS|<0.10` at every σ<0 candidate → routes to v10b.
* **Attribution disclaimer added**: v10a' is operational, not
  attribution-clean.
* **1e-14 framing corrected**: V=−0.10 probe only; bracket covers
  V=−0.30 + V=−0.50.

### Revised decision tree (verbatim replacement of the R2 version)

[See section 1 point 2 above.  Replaces the "Case A / Case B / Case C"
fragment from R2 with the seven-branch A–G version.]

### Revised threshold definitions

```markdown
## v10b-vs-v10c routing decision threshold (pinned)

After the v10a' run + (if needed) bracket sweep:

Route to **v10b** (literature-anchored Γ_max + k_des calibration)
iff EVERY σ_S<0, fallback-tier-valid V in the output satisfies:
  - `denominator_cap / denominator_total > 0.8`
  - `θ > 0.9`
  - `|dRnet_dsigma_along_stern_capacitance| < 0.10`

(All three conditions met means the Langmuir cap has eaten the
σ_S leverage; tuning kinetics won't help.)

Route to **v10c** (C_S bracket sweep) iff `abort_to_v10c=True`
(no V has σ_S<0).  Originally-planned v10c.

Route to **escalation** iff every bracket-sweep run produces an
artifact candidate (`locked_current_filter_passes_but_o2_transport_limited=True`)
or `no_candidate_passed_locked_rule`.  Acceptance bundle review.

Otherwise: route to **Phase A.2** at the returned V_kin.
```

## 3. Continued critique prompt

This is the third (final) round under the configured cap. Review
the updated plan and my responses. Push back on any defended-
poorly point. Raise any new issues. Re-issue any earlier issue
not adequately addressed. Same numbered format + verdict line:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Scope reminder: still narrow on the three subtle physics/parameter-
coupling points. Don't critique driver structure or test list.

If material blockers remain, please be explicit so the auto-revise
step can record them as Unresolved.
