# R4 — Counterreply

## Section 1: Acknowledgments

### Re your point 1 (Picard cap-iter could be valid convergence)

**Accept.** The classifier was too aggressive at length 8.

**Fix:** Refined classification:

```
def classify_picard_status(history, snes_converged):
    if not snes_converged:
        return "snes_failed"      # overrides everything
    n = len(history)
    if n == 0:
        return "no_iters"
    if n == 1:
        return "single_iter"      # pre-update γ not stored; can't verify
    last_rel = abs(history[-1] - history[-2]) / max(
        abs(history[-1]), abs(history[-2]), 1e-30
    )
    if n < 8:
        return "converged" if last_rel < 1e-4 else "early_break"
    # n == 8 (the iter cap)
    return "converged_at_iter_cap" if last_rel < 1e-4 else "iter_cap_hit_unconverged"
```

Routing eligibility for `k_hyd_route`:
`picard_status in {"converged", "converged_at_iter_cap", "single_iter"}`.
Other statuses are diagnostic failures.  `snes_failed` always wins.

### Re your point 2 (k_hyd_route requires only θ>0.9; could be upper-knee not plateau)

**Accept.**  At θ=0.9 the system isn't fully saturated; selectivity could
still drift with k_hyd.

**Fix:** Strengthen `k_hyd_route` saturation requirement and add a
local-slope gate:

```
k_hyd_route eligibility:
  * θ > 0.95                                      (stricter than 0.9)
  * d ln Γ / d ln k_hyd < 0.05                    (local saturation slope,
                                                   computed between this
                                                   k_hyd and the next-lower
                                                   k_hyd in the converged set)
  * o2_flux_levich_ratio < 0.9
  * picard_status ∈ {"converged", "converged_at_iter_cap", "single_iter"}
  * |mass_balance_residual_rel| < 5e-3
```

`k_hyd_route` = highest k_hyd in the converged set satisfying ALL of
the above.

The `θ > 0.9` value still appears as the COVERAGE soft-floor in
diagnostics ("first k_hyd to hit cap-coverage"), but is not part of
the route-eligibility test.

### Re your point 3 (deck target is a band, not a scalar)

**Accept.**  Acceptance bundle § primary criterion: "Per-cation max
RRDE-equivalent H₂O₂% in V_RHE window matches experimental value
within ±10 pp".  The K⁺ pH≈4 experimental band is `~25-50%`
(per `Cation Summary Table`).

**Fix:** Define `selectivity_gap_pp` as **interval distance**:

```
def selectivity_gap_pp(observed_pct, deck_band=(25.0, 50.0)):
    lo, hi = deck_band
    if observed_pct < lo:
        return lo - observed_pct          # positive = under band
    if observed_pct > hi:
        return hi - observed_pct          # negative = over band
    return 0.0                             # inside band
```

A.2 emits in JSON: `selectivity_observed_pct`, `selectivity_gap_pp`,
`selectivity_band` (recorded as `[25.0, 50.0]` so future runs can
update the band).  Routing test: `|selectivity_gap_pp| > 10` for
high-priority v10b (matches the acceptance bundle's ±10 pp tolerance).

### Re your point 4 (rung_diag two sources of truth)

**Accept.** Mutating `rung_diag` in place AND appending dicts means
`result.rungs` and `augmented_rungs` may both end up holding the
augmented data, OR diverge if the callback fails partway.

**Fix:** Use callback side-channel as the **sole source of truth**;
ignore `result.rungs` for output construction:

```python
augmented_rungs = []     # closure list; sole source of truth.
partial_rungs = []       # partial-failure capture.

def _rung_callback(scale, ok, ctx, rung_diag):
    # Make a defensive shallow copy so further mutations to rung_diag
    # by the solver don't leak into our snapshot.
    snapshot = dict(rung_diag)
    if not ok:
        partial_rungs.append(snapshot)
        return
    # Augment the snapshot (NOT rung_diag itself).
    snapshot["cd_mA_cm2"] = ...
    snapshot["pc_mA_cm2"] = ...
    snapshot["x_2e"] = ...
    snapshot["o2_flux_levich_ratio"] = ...
    snapshot["picard_status"] = classify_picard_status(
        snapshot.get("gamma_picard_history", []),
        snapshot.get("snes_converged", False),
    )
    snapshot["mass_balance_residual_rel"] = ...
    augmented_rungs.append(snapshot)

# After the call:
result = solve_lambda_ramp_from_warm_start(..., rung_callback=_rung_callback)
record["rungs"] = augmented_rungs       # NOT result.rungs
record["partial_rungs"] = partial_rungs
```

The plan documents that `result.rungs` is intentionally NOT consumed
in A.2.

### Re your point 5 (exception_phase mechanism not specified)

**Accept.** No structured phase enum exists in `LadderExhausted`.

**Fix:** Driver classifies `exception_phase` from a combination of
callback firing + exception message:

```python
fired_callback = len(augmented_rungs) > 0 or len(partial_rungs) > 0

try:
    result = solve_lambda_ramp_from_warm_start(...)
    exception_phase = None
except LadderExhausted as exc:
    msg = str(exc)
    if "warm-start SS re-converge failed" in msg:
        exception_phase = "warm_reconverge"
    elif "λ=0 floor solve failed" in msg:
        exception_phase = "lambda_zero"
    elif fired_callback:
        exception_phase = "lambda_positive"
    else:
        # Defensive: callback never fired AND message didn't match the
        # known prefixes; treat as ambiguous.
        exception_phase = "unknown"
    record["exception"] = msg
    record["exception_phase"] = exception_phase
```

A unit test asserts the three known message prefixes match the
solver's actual exception text (so we catch any solver-side string
drift in CI rather than at runtime).

### Re your point 6 ("no k_hyd_route → V_kin re-selection" overloads causes)

**Accept.** A single overloaded reroute target obscures the actual fix.

**Fix:** Decompose into 5 mutually-exclusive failure categories,
checked in order:

```
def classify_no_route_cause(per_k_hyd_records):
    # Inspect the rungs across the k_hyd grid to determine which
    # gate eliminated all candidates.
    saturated_set = [r for r in records if max(r.theta) > 0.9]
    saturated_clean_set = [
        r for r in saturated_set
        if r.picard_status_ok and r.mass_balance_ok
    ]
    saturated_clean_transport_ok_set = [
        r for r in saturated_clean_set
        if r.o2_flux_levich < 0.9
    ]

    if len(saturated_set) == 0:
        return "no_saturated_rung"        # cap never engaged in grid
    if len(saturated_clean_set) == 0:
        if any(not r.picard_status_ok for r in saturated_set):
            return "picard_failure"
        return "mass_balance_failure"
    if len(saturated_clean_transport_ok_set) == 0:
        return "transport_only"           # all saturated rungs are transport-limited
    if not satisfies_local_slope_gate(saturated_clean_transport_ok_set):
        return "grid_gap"                 # k_hyd grid too sparse near plateau
    # Should not reach here if k_hyd_route is None.
    return "ambiguous"
```

Routing per cause:
* **`no_saturated_rung`** → extend k_hyd grid upward (e.g., add 1e0,
  1e1) and re-run A.2.  Smoke Γ_max may be too generous; v10b
  literature anchor expected to shrink it.
* **`picard_failure`** → debug the Picard outer loop OR escalate to
  B.2's patched λ ladder ahead of schedule.  NOT a V_kin re-selection.
* **`mass_balance_failure`** → debug the residual-side closure
  (proton boundary source vs. closed-form Γ_ss).  NOT a V_kin
  re-selection.
* **`transport_only`** → V_kin re-selection (smaller K0_R4e_factor
  to drop O₂ consumption, OR pick a less cathodic V_kin).  This IS
  the "V_kin re-selection" path from R3.
* **`grid_gap`** → add k_hyd points between the saturated and
  pre-saturation regimes (e.g., {2e-2, 5e-2, 1e-1}).  Local fix.

### Re your point 7 ("span θ ∈ [0.10, 0.93]" ambiguous)

**Accept.**  "Span" can mean three different tests.

**Fix:** Use your suggested exact logic verbatim:

```
transition_coverage_ok = (
    min(theta_converged_at_lambda1) <= 0.10
    AND max(theta_converged_at_lambda1) >= 0.93
    AND len(transition_grid_converged_at_lambda1) >= 4
)
```

where `transition_grid = {1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}`
and `theta_converged_at_lambda1` is the θ value at the highest λ rung
(target 1.0; flag if < 1.0 due to AdaptiveLadder exhaustion).

---

## Section 2: Updated artifact (R4 deltas)

Overlaid on R3:

* **Driver §1 (R4-#5):** `exception_phase` classification from message
  + callback-fired flag.  Unit test asserts exception message prefixes
  match the solver's actual text.
* **Driver §1 (R4-#4):** rung augmentation uses defensive shallow copy;
  `augmented_rungs` is sole source of truth; `result.rungs` ignored.
* **Pass criterion (R4-#7):** "transition coverage" defined as exact
  `min ≤ 0.10 AND max ≥ 0.93 AND len ≥ 4`.
* **Routing §7 (R4-#2):** `k_hyd_route` requires `θ > 0.95` AND
  `d ln Γ / d ln k_hyd < 0.05` AND transport gate AND Picard OK AND
  mass-balance OK.
* **Routing §7 (R4-#3):** `selectivity_gap_pp` is interval distance to
  band `[25, 50]`; positive = under-band; negative = over-band; zero
  inside band.
* **Routing §7 (R4-#6):** if no `k_hyd_route`, classify cause as one of
  `{no_saturated_rung, picard_failure, mass_balance_failure,
  transport_only, grid_gap}` and route per the table above.
* **Picard classifier (R4-#1):** 6 statuses, with `converged_at_iter_cap`
  treated as success and `snes_failed` overriding all Picard statuses.

---

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
