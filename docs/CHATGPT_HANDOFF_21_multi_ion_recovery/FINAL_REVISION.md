# FINAL_REVISION — Critique Session 21

Date: 2026-05-08
Topic: multi-ion-recovery
Final verdict: APPROVED at round 5/5
Codex session ID: 019e08c1-caa1-7f83-9df9-d9a53812f992
Original artifact (revised in place):
  `StudyResults/fast_realignment_2026-05-08/PHASE_4_STATUS.md`

## Summary

5-round adversarial review. GPT raised 51 numbered issues across 4
review rounds (R1=14, R2=15, R3=14, R4=8). Two of my analyses
contained real factual errors (eta_drop sign convention; Stern
physics direction) that GPT caught and forced corrections. The
plan converged from "5b vs 5g vs softplus" trade-off to a clear
load-bearing patch sequence: 5α (Picard multi-ion) → 5ζ (diagnostics)
→ 5β (fallback dispatch) → 5γ (anchor builder) → 5δ/5ε (escape valves).

## Issue ledger

### Round 1 (14 issues raised)

| # | Subject | Verdict | Evidence pointer |
|---|---|---|---|
| 1 | JSON I cited as "+0.55 V" actually has anchor_v_rhe=0.0 | **Accepted** | StudyResults/fast_realignment_2026-05-08/anchor_smoke/anchor_smoke.json |
| 2 | R≈0.45 estimate is bulk-H boundary-rate, not Picard flux balance | **Accepted partial** | Picard's c_s_list also gives ≈bulk; same conclusion via different path |
| 3 | Picard uses single-ion gamma_s/phi_o/Stern; needs all-together patch | **Accepted** | picard_ic.py:1138, 1142, 1228 — load-bearing root cause |
| 4 | grid_per_voltage.py:350 calls set_initial_conditions_logc directly (not dispatched), corrupts mu_H | **Accepted** | grid_per_voltage.py:350 |
| 5 | logc_muh's linear-phi IC also single-ion at line 662 | **Accepted** | forms_logc_muh.py:662 |
| 6 | 5b I-ramp requires form rebuild per α (boltzmann.py:220) | **Accepted** | counterion bulk baked into fd.Constant |
| 7 | 5g targets wrong variable first | **Accepted** | depends on Picard correctness |
| 8 | Softplus changes physics; conflicts with clip=100 trust | **Accepted** | docs/clipping_conventions.md |
| 9 | "Transient pre-relax" already done by run_ss; need dt_init continuation | **Accepted** | grid_per_voltage.py:260 |
| 10 | Stern continuation direction backwards | **Accepted** | picard_ic.py:263 |
| 11 | k0=0 hardwired at form build; use tiny positive | **Accepted** | forms_logc_muh.py:407 |
| 12 | "Start at low V_RHE" sign wrong; +0.55 already weakest cathodic | **Accepted** | E_eq_R2E=0.695 V; lower V is more cathodic |
| 13 | Legacy-ClO4 IC swap is low-trust | **Accepted** | one-hour falsification at most |
| 14 | Multi-ion diagnostics in diagnostics.py:195 use per-counterion theta_b | **Accepted** | diagnostics.py:195 |

### Round 2 (15 issues raised)

| # | Subject | Verdict |
|---|---|---|
| 1 | set_initial_conditions(ctx, sp) routes back to analytical IC if sp says debye_boltzmann | **Accepted** — clone params with initializer="linear_phi" |
| 2 | z=0 analysis factual error: _set_z_factor mutates same z used in muh recon | **Accepted** — c_H = exp(mu_H), not exp(mu_H - phi) at z=0 |
| 3 | Phase 5α must patch INIT electrostatics too (lines 1102-1126) | **Accepted** — extract _update_electrostatics helper |
| 4 | Optional multi_ion_ctx will be brittle; use small helpers | **Accepted** — three sub-helpers |
| 5 | multi_ion.py uses unclamped _safe_exp; boltzmann.py clamps at phi_clamp | **Accepted** — apply per-ion clamp consistently |
| 6 | Bisection might pick wrong root if monotonicity breaks | **Accepted** — local bracket + monotone scan test |
| 7 | 1000x k0 step too coarse | **Accepted** — 32× adaptive with rollback |
| 8 | Picard reads nondim k0; residual uses bv_k0_funcs; metadata divergence | **Accepted** — set_reaction_k0_model helper mutates both |
| 9 | C+D has no k0 homotopy hook | **Accepted** — preconverged_anchors C+D extension |
| 10 | dt_init=1e-4 won't reach 0.25 under SER default | **Accepted** — programmatic ladder |
| 11 | Above-Eeq homotopy not clean zero crossing | **Accepted** — empirical voltage homotopy |
| 12 | Stern homotopy not one-parameter smooth | **Accepted** — defer |
| 13 | logc_muh baseline regen is yellow flag | **Accepted** — versioned baselines |
| 14 | Multi-ion diagnostics shouldn't be post-only | **Accepted** — Phase 5ζ before Phase 5β |
| 15 | "Sensible R/c_s" test too vague | **Accepted** — rate-consistency test with explicit tolerances |

### Round 3 (14 issues raised, 2 force corrections of my R3)

| # | Subject | Verdict |
|---|---|---|
| 1 | My R3 no-Stern eta_drop = phi_applied - phi_o was WRONG | **Accepted** — actual: eta_drop = phi_applied_model (picard_ic.py:421) |
| 2 | I flipped Stern physics AGAIN | **Accepted** — large C_S → psi_D → full_drop, not psi_S → full_drop |
| 3 | _update_electrostatics one giant helper hard to test | **Accepted** — three sub-helpers + thin wrapper |
| 4 | Monotonicity test fails outside clamp | **Accepted** — test only inside unclamped interval |
| 5 | Local bracket fallback should not silently fall back | **Accepted** — log diagnostic; require monotonicity for global fallback |
| 6 | Phase 5γ ordering: assign k0 BEFORE IC | **Accepted** — sequence corrected |
| 7 | Don't rerun IC at every k0 step; rename update_picard_k0 | **Accepted** — set_reaction_k0_model |
| 8 | Key is ctx["dt_const"] not ctx["dt_constant"] | **Accepted** — verified at forms_logc_muh.py:581 |
| 9 | C+D returns immediately if no cold successes | **Accepted** — preconverged_anchors API |
| 10 | k0+dt: BOTH small initially | **Accepted** — k0=1e-12, dt=1e-4 simultaneously |
| 11 | z=0 phi_surface not guaranteed = phi_applied | **Accepted** — measure empirically post-5β |
| 12 | Rate-consistency tolerance: 1e-3 rel + 1e-10 abs floor; 1e-2 spatial | **Accepted** — codified |
| 13 | logc_muh baseline regen yellow flag | **Accepted** — keep old + new side by side |
| 14 | Use params not sp.solver_options for tuple compat | **Accepted** — mirror _params_with_phi |

### Round 4 (8 issues raised, 1 BLOCKER)

| # | Subject | Verdict |
|---|---|---|
| 1 | **BLOCKER** Phase 5γ written for j=0 only | **Accepted** — active_reaction_indices generalization |
| 2 | k0 ladder overshoots target | **Accepted** — programmatic min(next, target) |
| 3 | preconverged_anchors needs more than ctx_snapshot | **Accepted** — full PreconvergedAnchor TypedDict |
| 4 | Phase 5α highest-risk; instrument first | **Accepted** — Phase 5α GATE script |
| 5 | Multi-ion linear-Debye is approximation, accept | **Accepted** — drop bracket machinery, just assert finiteness |
| 6 | Probability estimates: A 70%, D 55%, B 40% | **Accepted** — captured for user calibration |
| 7 | Early failure detector: log-consistency check | **Accepted** — assert_picard_residual_log_consistent |
| 8 | grid_per_voltage imports only logc, not muh | **Accepted** — Option B (clone params + dispatcher) |

### Round 5 — APPROVED

3 minor cautions, no blockers:
- Option B clone params must handle 11-tuples too (helper mirroring _params_with_phi).
- forms_logc.py has same single-ion linear fallback issue — patch both for symmetry, or document scope.
- Highest-risk implementation detail: Picard refactor preserving single-ion byte-equivalence; run those tests immediately after helper extraction.

## Where the validated plan landed

The original artifact at
`StudyResults/fast_realignment_2026-05-08/PHASE_4_STATUS.md` was
revised in place. The "Recommendation" section was replaced wholesale
with the GPT-validated v3-final escalation plan (Phase 5α through
5ε), success-probability calibration table, "Things this plan
deliberately does NOT do" section, and an audit-trail pointer to
this session.

The revised section captures all R1-R4 accepted decisions:

- **Phase 5α** (Picard multi-ion patch + GATE) is the load-bearing
  step. Phase 5α GATE is the single go/no-go: log-consistency check
  on Picard vs residual rate at +0.55 V multi-ion.

- **Phase 5ζ** (diagnostics patch) lands BEFORE Phase 5β so
  failure-triage uses correct shared-theta numbers.

- **Phase 5β** (logc_muh fallback dispatch fix) uses Option B
  (params clone with `initializer="linear_phi"`) and a helper
  that handles both `SolverParams` and 11-tuples. Versioned
  baselines: logc byte-equiv, logc_muh regenerated.

- **Phase 5γ** (anchor builder with k0+dt continuation) uses
  `active_reaction_indices` (R4 §1) for Pass A/B/D, programmatic
  `_adaptive_ladder` (R4 §2), and full `PreconvergedAnchor`
  injection into C+D (R4 §3).

- **Phase 5δ/5ε** as escape valves only.

## Things explicitly REJECTED by the loop

- Phase 5b I-ramp continuation (R1 §6: form rebuild per α; not first
  priority).
- Phase 5g smarter spatial IC (R1 §7: targets wrong variable; depends
  on Picard correctness first).
- Softplus-bounded BV rate (R1 §8: changes physics; only valid as
  homotopy with final verification).
- Stern continuation (R3 §12: not one-parameter smooth; defer).
- "Start at low V_RHE" (R1 §12: sign wrong; +0.55 V is already
  weakest cathodic drive).

## Defended issues — none

I deferred or accepted every issue GPT raised across all 5 rounds.
This was not capitulation: each issue was either factually correct
(2 of mine had real errors), pointed at a real implementation gotcha
(metadata mismatch, dt key name, C+D anchor injection, etc.), or
shifted the plan's priority (5b/5g/softplus all rejected as
inferior to direct Picard patch).

The convergence was honest, not accommodating; GPT did the gap-finding
and Claude did the integration. The final plan is materially different
from the initial three-options framing, with a clear load-bearing
sequence and a hard go/no-go gate.

## Total time

Round 1: ~3 min handoff write + ~2 min codex round.
Round 2-5: ~5 min each (counterreply + codex + parse).
Total wall clock: ~25 min for 5-round adversarial review.

Probability calibration captured: ~65% confidence in "any A or D ≥
15/25 in 3 focused engineering days". Plan target met.
