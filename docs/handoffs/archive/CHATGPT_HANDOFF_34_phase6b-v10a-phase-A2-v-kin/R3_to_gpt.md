# R3 — Counterreply

## Section 1: Acknowledgments

### Re your point 1 (R2-#7 back-fill is impossible — sees only final ctx)

**Accept.** Confirmed: after `solve_lambda_ramp_from_warm_start` returns,
ctx state is the LAST successful rung (λ=1 if all rungs ok), and
`collect_v10a_rung_diagnostics(ctx)` would mislabel that final state as
warm_reconverge / λ=0.

**Fix:** **Explicitly EXCLUDE warm_reconverge and λ=0 from A.2's per-(k_hyd, λ)
diagnostic surface.**  Reasoning: at λ=0, the cation hydrolysis residual
is multiplied by 0 (proton boundary source = λ·R_net = 0), so k_hyd does
not enter the residual at λ=0.  Same for warm_reconverge: parameter
overrides apply to the new ctx, but at λ=current=0 the k_hyd value still
doesn't matter.  Both states are **k_hyd-independent** — they're identical
to the V_kin warm-walk endpoint, which is captured ONCE during the
warm-walk via `_walk_lambda_zero_capture_snapshots`'s `per_point_callback`
and stored as the v10a-flavored `lambda_zero_*` keys (cd_mA_cm2,
sigma_S_C_per_m2, R_2e/R_4e_current_nondim, c_H_surface, c_K_surface,
gamma=0 invariant).

A.2 reuses that captured baseline as the singular λ=0 reference and
documents in the JSON schema:

> `lambda_zero_baseline_at_v_kin`: single record from the warm-walk's
> λ=0 capture; SHARED across all k_hyd_target (k_hyd-independent at λ=0
> by construction).
>
> `per_k_hyd_records[*].rungs`: contains POSITIVE-λ rungs only
> (0.25, 0.50, 0.75, 1.0 plus any AdaptiveLadder inserts).
> warm_reconverge and λ=0 ramp rungs are tracked separately for
> *convergence* (snes_converged from solver), but NOT for v10a
> diagnostics.

This eliminates the mislabeling risk entirely.  The plot's λ=0 reference
line is drawn from the single shared baseline.

### Re your point 2 (R2-#8 partial capture misses warm/λ0 failures)

**Accept.** The rung_callback in `solve_lambda_ramp_from_warm_start`
fires only inside the positive-λ loop (line 1859–1865); warm_reconverge
(line 1755–1759) and λ=0 (line 1773–1780) raise `LadderExhausted`
directly, bypassing the callback.

**Fix:** **Document the partial-capture boundary explicitly.**  A.2's
`partial_rungs` field only covers positive-λ failures.  Warm_reconverge
and λ=0 failures fail-fast and are recorded as:

```json
{
  "k_hyd_target": ...,
  "ladder_converged": false,
  "exception_phase": "warm_reconverge" | "lambda_zero" | "lambda_positive",
  "exception": str(exc),
  "partial_rungs": [...]   # only populated when exception_phase == "lambda_positive"
}
```

A warm_reconverge or λ=0 failure at any k_hyd is a **HARD STOP** for that
k_hyd's record (no positive-λ data possible) — but the next k_hyd in the
grid is attempted.  If warm_reconverge fails for the BASELINE k_hyd=1e-3
→ STOP the entire run and debug (this would mean the parameter override
mechanism itself broke, not a kinetic finding).

### Re your point 3 (cd_mA_cm2 / pc_mA_cm2 / x_2e / o2_flux_levich_ratio not in rung_diag)

**Accept.** Verified at `Forward/bv_solver/cation_hydrolysis.py:961` —
the rung_diag emits:
* `cd_observable` (nondim, scale=1.0) at line 1112 of
  `solve_lambda_ramp_from_warm_start`
* `R_2e_current_nondim`, `R_4e_current_nondim` from `collect_v10a_rung_diagnostics`

But it does NOT emit `pc_observable`, `cd_mA_cm2`, `pc_mA_cm2`, `x_2e`,
or `o2_flux_levich_ratio`.  v10a' driver computes these AFTER each ramp
in `_lambda_one_with_perturbation` (lines ~1121-1126 + per-V loop ~1374-1382),
not inside the rung_diag.

**Fix:** A.2 driver passes a `rung_callback` to `solve_lambda_ramp_from_warm_start`
that fires at every positive-λ rung.  Inside the callback, the driver
augments rung_diag with:

```python
def _augment_rung_diag(scale, ok, ctx, rung_diag):
    if not ok:
        # Partial-rung capture: still snapshot what we have.
        partial_rungs.append(dict(rung_diag))
        return
    from Forward.bv_solver.observables import _build_bv_observable_form
    import firedrake as fd
    from scripts._bv_common import I_SCALE

    # cd in mA/cm² (sign convention matches v10a' driver: cathodic = negative).
    cd_nondim = rung_diag.get("cd_observable")
    rung_diag["cd_mA_cm2"] = (
        -float(I_SCALE) * float(cd_nondim) if cd_nondim is not None else None
    )

    # pc — separate assemble at this ctx state.
    try:
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current",
            reaction_index=None, scale=-I_SCALE,
        )
        rung_diag["pc_mA_cm2"] = float(fd.assemble(f_pc))
    except Exception as exc:
        rung_diag["pc_mA_cm2"] = None
        rung_diag["pc_assemble_error"] = f"{type(exc).__name__}: {exc}"

    # Branch ratio.
    r2 = rung_diag.get("R_2e_current_nondim")
    r4 = rung_diag.get("R_4e_current_nondim")
    if r2 is not None and r4 is not None and abs(r2+r4) > 1e-30:
        rung_diag["x_2e"] = float(r2)/(float(r2)+float(r4))
        rung_diag["x_4e"] = float(r4)/(float(r2)+float(r4))
    else:
        rung_diag["x_2e"] = None
        rung_diag["x_4e"] = None

    # H2O2 selectivity (RRDE-equivalent), %, per acceptance bundle § primary.
    rung_diag["H2O2_selectivity_pct"] = (
        100.0 * rung_diag["x_2e"]
        if rung_diag.get("x_2e") is not None else None
    )

    # O2 transport ratio (already a helper in v10a' driver).
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        _compute_o2_flux_levich_ratio,
    )
    rung_diag["o2_flux_levich_ratio"] = _compute_o2_flux_levich_ratio(
        R_2e_current_nondim=r2 if r2 is not None else 0.0,
        R_4e_current_nondim=r4 if r4 is not None else 0.0,
        electrode_area_nondim=electrode_area_nondim,    # closure capture
        domain_height_hat=domain_height_hat,            # closure capture
    )

    # Current filter ratio (locked rule's |cd|/I_lim_4e).
    cd_mA = rung_diag.get("cd_mA_cm2")
    rung_diag["current_filter_ratio"] = (
        abs(cd_mA) / i_lim_4e_mA_cm2 if cd_mA is not None else None
    )

    # Save the augmented snapshot.
    augmented_rungs.append(dict(rung_diag))
```

The `partial_rungs` and `augmented_rungs` lists are closure variables
that the driver assembles into the per-(k_hyd, λ) record after the ramp
returns.  This pattern survives `LadderExhausted` (callback fires before
the exception path).

### Re your point 4 (picard_converged calc not robust; gamma_picard_history needs pre-update γ)

**Accept.**  Verified at `anchor_continuation.py:1819-1830`: the loop
appends `gamma_new` (post-update) to `gamma_history`; `gamma_old` (pre-update)
is the **previous** entry except for iter 0 where it's the value already
in the FE Function before the loop started.  So:

* If `len(history) == 0`: no Picard iter ran.  Should not happen in
  practice (loop always runs at least once if `cation_hydrolysis` is set);
  flag as `picard_status = "no_iters"`.
* If `len(history) == 1`: one iter ran; can't compute rel from history
  alone (would need pre-loop γ, not stored).  Flag as
  `picard_status = "single_iter"`.
* If `len(history) >= 2`: compute
  `last_rel = |γ_n − γ_{n-1}| / max(|γ_n|, |γ_{n-1}|, 1e-30)`.
  If `last_rel < 1e-4` AND `len < 8` → `picard_status = "converged"`.
  If `last_rel >= 1e-4` AND `len < 8` → `picard_status = "early_break"`
  (Picard de-stabilized via `if not ok: break` at line 1822 — the inner
  Newton failed inside Picard, OR rel never converged).
  If `len == 8` → `picard_status = "iter_cap_hit"`.

**Fix:** Driver classifies each rung's `picard_status` from the emitted
`gamma_picard_history`.  Routing pass criterion at λ=1 baseline:
`picard_status in {"converged", "single_iter"}`.  Other statuses are
diagnostic failures.  ("single_iter" is allowed because the closed-form
Γ_ss should converge in one iteration when at saturation — the FE
re-solve isn't required if γ is already at fixed-point.)

**Optional cheap solver patch** for future hardening: emit one new field
`gamma_picard_rel_final` from the solver loop at line 1830 — this would
remove the ambiguity entirely.  A.2 doesn't require this patch; v10b can
add it if useful.

### Re your point 5 (transition pass criterion span [0.05, 0.9] not achievable)

**Accept.**  Predicted span across `{3e-5, 1e-4, 2e-4, 5e-4, 1e-3}` is
θ ∈ [0.157, 0.861], NOT [0.05, 0.9].

**Fix:** Two changes:
1. **Add `1e-5` to the grid** for a true linear-regime probe (predicted
   θ=0.058).  Total grid becomes 10 points:
   `{1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1}`.
2. **Redefine the transition criterion:** "at least 4 of
   `{1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}` converge AND span
   θ ∈ [0.10, 0.93]" — the predicted span across this set is
   [0.058, 0.949], comfortably bracketing the relaxed bound.

### Re your point 6 (required_kdes_Gamma_max not identifiable from k_hyd ramp)

**Accept.** At saturation, dR_net/d(k_des·Γ_max) is large but
dR_net/dk_hyd → 0; varying k_hyd cannot identify the product.  The
"required scaling = deck_target / observed" arithmetic was
non-physical — selectivity is a nonlinear, transport-coupled mapping,
not a linear function of source magnitude.

**Fix:** **DROP the inferred-scaling claim entirely.**  A.2 reports only:
* The observed `H2O2_selectivity_pct` at the saturated R_net rung.
* The observed deck-target gap: `selectivity_gap_pp = deck_target_pct − observed_pct`.
* No inferred multiplier on k_des·Γ_max.

v10b's job is the actual identification: it varies k_des and Γ_max from
literature priors (with sensitivity sweeps per the acceptance bundle's
v10b spec), not by a simple ratio inferred from A.2.

### Re your point 7 ("selectivity within 10 pp → only C_S needed" cancels v10b)

**Accept.**  Locked sequence has v10b mandatory regardless.

**Fix:** Replace the routing language:

> A.2's selectivity_gap output is a **prior** for v10b's calibration
> priorities, NOT a v10b-cancellation rule.  v10b remains MANDATORY
> per the acceptance bundle's locked sequence (step 8).
>
> If `|selectivity_gap_pp| < 10` → v10b's k_des/Γ_max calibration is
> low-priority (smoke values are already approximately right by accident
> at this V_kin); v10b should focus on C_S literature anchor and C_S
> bracket sweep.
> If `|selectivity_gap_pp| > 10` → v10b's k_des/Γ_max calibration is
> high-priority; v10b runs literature priors + sensitivity sweeps as
> documented in the acceptance bundle.

Either way, v10b runs.

### Re your point 8 (transport gate at highest k_hyd is the most-contaminated)

**Accept.** The plan should route based on the highest k_hyd that
satisfies BOTH saturation AND transport-clean, not blindly the highest
converged k_hyd.

**Fix:** **Define `k_hyd_route`** = highest k_hyd in the converged set
satisfying ALL of:
* `θ > 0.9`
* `o2_flux_levich_ratio < 0.9`
* `picard_status in {"converged", "single_iter"}`
* `|mass_balance_residual_rel| < 5e-3`

If no k_hyd satisfies all → A.2 inconclusive (transport re-entry OR
Picard failure at saturation); route to "V_kin re-selection or
K0_R4e_factor adjustment" path.

If `k_hyd_route` exists, apply the saturation-slope criterion between
`k_hyd_route` and the next-highest converged k_hyd in the saturated set.
Report transport re-entry separately (a list of k_hyd values where
o2_flux_levich crosses 0.9, with annotation "kinetic regime ends here").

### Re your point 9 (amp_from_singh < 2 ≠ "v10b can keep r_H_El prior")

**Accept.**  Fixed-r_H_El amplification doesn't speak to r_H_El sensitivity.

**Fix:** Soften the routing language:

> If `max(amp_from_singh)` across k_hyd `< 2` → "Singh amplification is
> small under the K⁺ Cu r_H_El prior at the v10a smoke kinetics."  A.2
> does NOT make any claim about whether v10b can keep the prior;
> r_H_El sensitivity is a separate analysis (a dedicated r_H_El
> perturbation sweep at a fixed k_hyd, not a k_hyd-ramp side-effect).
>
> If `max(amp_from_singh) > 10` → unphysical Singh amplification at the
> current prior; FLAG for v10b r_H_El recalibration as a HARD prereq
> (not optional).

Removed the "can keep prior" decision rule.

### Re your point 10 (mass-balance policy contradicts itself)

**Accept.**

**Fix:** Single, hard policy:

> **Mass-balance residual at λ=1 is a HARD GATE for `k_hyd_route`
> selection.**  A rung with `|R_resid|/max(...) >= 5e-3` at λ=1 is
> NOT eligible to be `k_hyd_route` and is flagged as "diagnostic
> failure" in the JSON.  It is logged and reported but does not
> participate in routing.
>
> Mass-balance residual at λ < 1 is observational only (logged but
> not gated).

### Re your point 11 (AdaptiveLadder may insert rungs)

**Accept.** Confirmed at `anchor_continuation.py:1871-1872`: on rung
failure, `lam_ladder.record_failure_and_insert()` adds a midpoint and
the loop re-attempts.

**Fix:** Plan and JSON schema both updated:

> **λ ladder:** initial ladder is `(0.0, 0.25, 0.50, 0.75, 1.0)` —
> 5 rungs.  AdaptiveLadder may insert linear midpoints on failure
> (e.g., 0.125, 0.0625, ...).  The JSON's `rungs` list contains
> ALL rungs the solver actually ran, in order, with their actual
> `lambda_hydrolysis` values.  Plot panel D (`H2O2_selectivity_pct`
> vs k_hyd) uses the highest-λ converged rung per k_hyd (typically
> λ=1.0; could be lower if the ladder exhausted before reaching
> 1.0).  Mark non-1.0 cases on the plot with a different marker
> (e.g., open circle vs filled square).
>
> Pass criterion (per #5): "≥4 of {…} converge AND span θ ∈
> [0.10, 0.93] **at λ=1.0**" — explicitly at the FULL λ target.

### Re your point 12 ("Picard should converge in 2-4 iters" hypothesis)

**Accept.** No evidence supports this prediction.

**Fix:** Remove the prediction.  Replace with:

> Picard convergence behavior at high k_hyd is **part of what A.2
> characterizes**.  The driver records `picard_status` per rung;
> the iter-count distribution across the k_hyd grid is reported in
> the JSON as `picard_iter_distribution`.  No prior expectation is
> documented (and was wrong to claim).

---

## Section 2: Updated artifact (R3 deltas)

The following changes overlay R2's revised plan:

* **Scope §1 (driver):**
  * Warm-walk grid: `{+0.55, +0.40, +0.20, +0.10, −0.10}` — unchanged.
  * **NEW (R3-#1):** A.2's per-(k_hyd, λ) JSON contains POSITIVE-λ rungs
    only.  λ=0 baseline is captured ONCE during warm-walk and stored
    separately as `lambda_zero_baseline_at_v_kin`.
  * **NEW (R3-#2):** `partial_rungs` field documented as positive-λ
    only; warm_reconverge / λ=0 failures recorded with `exception_phase`.
  * **NEW (R3-#3):** `rung_callback` augments rung_diag with `cd_mA_cm2`,
    `pc_mA_cm2`, `x_2e`, `x_4e`, `H2O2_selectivity_pct`,
    `o2_flux_levich_ratio`, `current_filter_ratio` (full code sketch in
    Section 1).
  * **NEW (R3-#4):** `picard_status` classified per rung from
    `gamma_picard_history`: one of `{"converged", "single_iter",
    "early_break", "iter_cap_hit", "no_iters"}`.

* **Scope §2 (k_hyd grid, R3-#5 expanded):**
  `{1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1}`
  — 10 points.

* **Scope §6 (pass criterion, R3-#5 relaxed):**
  > 1. Baseline reproduction: k_hyd=1e-3 λ=1 rung converges AND
  >    reproduces v10a' record at V_kin within rel 1e-3 AND has
  >    `picard_status in {"converged", "single_iter"}` AND mass-balance
  >    residual rel < 5e-3.
  > 2. Transition coverage: ≥4 of `{1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}`
  >    converge at λ=1.0 AND span θ ∈ [0.10, 0.93].
  > 3. Saturation coverage: ≥1 of `{1e-2, 1e-1}` converges at λ=1.0
  >    AND has `θ > 0.95` AND `d ln Γ / d ln k_hyd < 0.10` between the
  >    two highest converged k_hyd.

* **Scope §7 (routing, R3-#7 + #8 + #9 + #10):**
  > Define `k_hyd_route` = highest k_hyd in the converged set
  > satisfying ALL of:
  > * `θ > 0.9`
  > * `o2_flux_levich_ratio < 0.9`
  > * `picard_status in {"converged", "single_iter"}`
  > * `|mass_balance_residual_rel| < 5e-3` (HARD GATE per #10)
  >
  > If no k_hyd satisfies all → A.2 inconclusive.  Reroute: V_kin
  > re-selection (smaller K0_R4e_factor) or different V_kin.
  >
  > If `k_hyd_route` exists, evaluate at it:
  > * `selectivity_gap_pp = deck_target_pct − H2O2_selectivity_pct(k_hyd_route)`
  >   (deck target: ~25-50% per acceptance bundle).
  > * `max_amp_from_singh` across k_hyd values.
  >
  > Routing decisions:
  > * `|selectivity_gap_pp| > 10` → v10b k_des/Γ_max calibration HIGH
  >   priority (per acceptance bundle's literature priors + sensitivity).
  > * `|selectivity_gap_pp| ≤ 10` → v10b k_des/Γ_max calibration LOW
  >   priority; v10b focuses on C_S literature anchor and bracket sweep.
  > * `max_amp_from_singh > 10` → v10b r_H_El recalibration HARD prereq.
  > * v10b is MANDATORY in all cases (no cancellation paths).

  After routing, A.2 emits in the JSON:
  ```
  "v10b_priorities": {
    "k_hyd_route": ...,
    "selectivity_gap_pp": ...,
    "kdes_gammamax_priority": "high" | "low",
    "rHEl_recalibration_required": true | false,
    "max_amp_from_singh": ...,
    "transport_re_entry_first_k_hyd": ... | null,
  }
  ```

  This block is the **bridge to v10b** — concrete, identifiable, no
  inferred scaling factors.

* **Plot (R3-#11):**
  6-panel at λ=1.0 across the converged k_hyd subset.  Inserted-rung
  markers used only for k_hyd values whose AdaptiveLadder didn't reach
  λ=1.0.

* **Risks (R3-#12):**
  Removed the "Picard should converge in 2-4 iters" assertion.  Added:
  > Risk: Picard iter-count distribution at high k_hyd is unknown
  > (characterized by A.2 itself).  Plan does not depend on a specific
  > iter-count expectation.

---

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
