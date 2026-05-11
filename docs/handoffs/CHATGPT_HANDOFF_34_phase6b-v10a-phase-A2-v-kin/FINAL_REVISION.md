# FINAL REVISION — Phase 6β v10a Phase A.2 plan

* **Path revised:** `/Users/jakeweinstein/.claude/plans/phase6b-v10a-phase-A2-v-kin.md`
* **Critique session:** 4 rounds, **APPROVED** at R4
* **GPT quote (R4):** *"These are implementation nits, not plan blockers.
  The major conceptual and routing holes from R1-R3 are now closed."*
* **Tally:** 41 issues total · 41 addressed · 0 defended · 0 unresolved

## Issue ledger (chronological, with disposition + landing pointer)

### Round 1 (18 issues)

| # | Issue | Disposition | Landed |
|---|---|---|---|
| R1.1 | Outer Γ Picard masks non-convergence | Accepted | `picard_status` classifier; `mass_balance_residual_rel` HARD GATE |
| R1.2 | k_hyd grid {1e-4 … 1e+1} jumps over cap-onset transition (~1.6e-4) | Accepted | New 10-point grid `{1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1}` |
| R1.3 | Predicted table arithmetic wrong at k_hyd=1e-4 | Accepted | Table re-derived; θ=0.382 not 0.10 at 1e-4 |
| R1.4 | `denom_cap/total ≡ θ` at λ=1; routing rule double-counts | Accepted | Replaced with cap-coverage + slope-based saturation gates |
| R1.5 | `|sensS|<0.10` not computable when perturbation skipped | Accepted | Dropped from default routing; opt-in via `--with-perturbation` at k_hyd_route only |
| R1.6 | v10b bridge under-specified | Accepted | `v10b_priorities` JSON block (no inferred scaling factors) |
| R1.7 | λ=0 + warm_reconverge missing v10a diagnostics | Accepted | Excluded from per-(k_hyd, λ) surface; single shared baseline from warm-walk |
| R1.8 | LadderExhausted loses partial rungs | Accepted | `rung_callback` side-channel captures `partial_rungs` |
| R1.9 | ≥5/6 pass criterion too lax | Accepted | Structured 3-criterion pass: baseline reproduction, transition span, saturation coverage |
| R1.10 | Warm-walk path under-specified | Accepted | Reuse v10a' grid `{+0.55, +0.40, +0.20, +0.10, -0.10}` |
| R1.11 | Fake 5-min hard-cap | Accepted | Removed; `max_ss_steps_per_rung=300` + SNES `max_it=400` |
| R1.12 | `R_net = k_des·Γ` tautological | Accepted | Replaced with assembled-side mass-balance residual |
| R1.13 | σ_S sign sanity overclaimed | Accepted | Demoted to observation; sign-bug detection deferred to plumbing ablation |
| R1.14 | x_2e branch sanity overclaimed | Accepted | Demoted to observation |
| R1.15 | Transport re-entry not guarded | Accepted | `o2_flux_levich_ratio < 0.9` is `k_hyd_route` HARD gate |
| R1.16 | r_H_El sensitivity claim mismatch | Accepted | Monitor existing `amp_from_singh(k_hyd)` curve; no new perturbation |
| R1.17 | K0_R4e_factor invariant for v10b | Accepted | HARD precondition documented in plan + CLAUDE.md + B.2 driver assertion |
| R1.18 | "Γ saturates BUT denom_cap<0.8" incoherent | Accepted | Subsumed by R1.4 fix (algebraic identity removes the branch) |

### Round 2 (12 issues)

| # | Issue | Disposition | Landed |
|---|---|---|---|
| R2.1 | R2-#7 back-fill impossible (sees only final ctx) | Accepted | EXCLUDE warm_reconverge + λ=0 from per-(k_hyd, λ) surface |
| R2.2 | Partial capture misses warm/λ0 failures | Accepted | Documented partial_rungs covers positive-λ only; `exception_phase` enum |
| R2.3 | `cd_mA_cm2` etc. NOT in rung_diag | Accepted | Custom rung_callback augments per rung |
| R2.4 | `picard_converged` calc not robust | Accepted | `classify_picard_status` with 6 statuses; `single_iter` defensible per current helper |
| R2.5 | Transition span [0.05, 0.9] not achievable | Accepted | Added 1e-5 to grid; redefined criterion as `min≤0.10 AND max≥0.93 AND len≥4` |
| R2.6 | `required_kdes_Gamma_max` not identifiable | Accepted | DROPPED inferred-scaling claim; report observed `single_v_selectivity_gap_pp` only |
| R2.7 | "Selectivity within 10pp → only C_S" cancels v10b | Accepted | v10b MANDATORY in all branches; A.2 informs priority, not cancellation |
| R2.8 | Transport gate at highest k_hyd is most-contaminated | Accepted | `k_hyd_route` requires saturation AND transport-clean simultaneously |
| R2.9 | `amp_from_singh<2` ≠ "v10b can keep prior" | Accepted | Softened: "Singh amplification small under current prior"; no decision |
| R2.10 | Mass-balance policy contradicts itself | Accepted | HARD GATE for `k_hyd_route`; observational at λ<1 |
| R2.11 | AdaptiveLadder may insert rungs | Accepted | Plan + JSON schema state initial-vs-actual ladder; arbitrary λ values supported |
| R2.12 | "Picard converges in 2-4 iters" hypothesis | Accepted | Removed; `picard_iter_distribution` reported in JSON, no prior expectation |

### Round 3 (7 issues)

| # | Issue | Disposition | Landed |
|---|---|---|---|
| R3.1 | Picard cap-iter could be valid converge | Accepted | `converged_at_iter_cap` status; `snes_failed` overrides |
| R3.2 | `k_hyd_route` requires only θ>0.9; could be upper-knee | Accepted | Stricter θ>0.95 + local slope `d ln Γ / d ln k_hyd < 0.05` |
| R3.3 | Deck target is a band, not scalar | Accepted | `single_v_selectivity_gap_pp` interval distance to `[25, 50]` |
| R3.4 | rung_diag two sources of truth | Accepted | callback side-channel sole source; `result.rungs` ignored |
| R3.5 | `exception_phase` mechanism not specified | Accepted | Driver classifies from message + callback-fired flag; unit test guards string drift |
| R3.6 | "no k_hyd_route" overloads 5 causes | Accepted | `classify_no_route_cause` enum: no_saturated, picard_failure, mass_balance_failure, transport_only, grid_gap |
| R3.7 | "Span" ambiguous | Accepted | Exact: `min ≤ 0.10 AND max ≥ 0.93 AND len ≥ 4` |

### Round 4 (4 nits)

| # | Issue | Disposition | Landed |
|---|---|---|---|
| R4.1 | `selectivity_gap_pp` is single-V proxy, not bundle pass/fail | Accepted | Renamed `single_v_selectivity_gap_pp`; labeled advisory |
| R4.2 | `classify_no_route_cause` should inspect only λ=1 records | Accepted | `lambda1_record(record)` helper |
| R4.3 | Callback exceptions silently dropped | Accepted | Callback wrapped in try/except; appends `callback_augment_error` always |
| R4.4 | `single_iter` Picard success defensible only via current helper control flow | Accepted | Comment in classifier tracks `anchor_continuation.py:1819-1830` |

## Defended (0)

None.  Every GPT issue was accepted; this matches the plan's intent
to prefer round-by-round honest engagement over reflexive defense.

## Unresolved (0)

None.  Loop converged at R4 with VERDICT: APPROVED.

## Pointer to revised artifact

The full revised plan is at:
`/Users/jakeweinstein/.claude/plans/phase6b-v10a-phase-A2-v-kin.md`

Key sections updated:
* `## Provenance` — new (cites this critique session)
* `## Scope (Phase A.2 only)` § 1-8 — full rewrite incorporating
  R1-R4 fixes
* `## Implementation notes` — new `classify_picard_status`,
  `_compute_mass_balance_residual_rel`, expected-behavior table
* `## Verification` — pass criterion + sanity checks per R3-R4
* `## Risks` — 16 risks (was 8 in R1)
* `## Critique provenance (session 34)` — new (this ledger summary)
