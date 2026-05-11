# FINAL_REVISION — Critique session 36

**Path revised:**
`/Users/jakeweinstein/.claude/plans/phase6b-step8-v10b-calibration.md`
(v1 → v7-FINAL).

**Final verdict:** APPROVED at round 7.
**Issue trajectory:** R1=20, R2=15, R3=9, R4=5, R5=3, R6=1, R7=0.
**Total accepted:** 53.  **Defended:** 0.  **Unresolved:** 0.

---

## Addressed (53 accepted issues, by round)

### Round 1 (20 issues, all accepted)

| # | Issue | Patch | Lands at |
|---|---|---|---|
| 1 | k_des detailed-balance rule wrong (Singh pKa = k_hyd/k_prot, NOT k_hyd/k_des) | Strategy 1 removed; Singh→k_prot audit moved to writeup §6 open ask | §3.3 |
| 2 | No nondim map for k_des | §3.0 added: `k_des_nondim = k_des_phys · τ_REF`, τ_REF≈5s | §3.0 |
| 3 | Solver default k_des not updated | D3 updates `raw_cfg.get("k_des", 1.0)` → V10B at line 293 | D3 |
| 4 | Step 6 A0 audit hardcoded to v10a A.2 JSON | D6 requires explicit `--a2-baseline-json` CLI flag | D6 |
| 5 | D6 R_net not in audit | Phase v10b.C step 1 verifies + adds R_net to audit keys | D6 |
| 6 | `SMOKE = V10B` aliases destroy provenance | D3/D4/D4': freeze V10A, introduce V10B, alias only `SMOKE = V10A` | D3, D4, D4' |
| 7 | Alias/export/test plan incomplete | D8 explicit tests for V10A/V10B coexistence + production-default | D8 |
| 8 | k_des bracket diagnostic contradicts itself | D7-D3 evaluates BOTH k_hyd_baseline AND k_hyd_route | D7-D3 |
| 9 | No coupled Γ_max × k_des sensitivity | D7-D4 added: 30-rung matrix | D7-D4 |
| 10 | D5 ±20% tolerance can reject valid calibration | D5 split into HARD (escalate) + SOFT (logged) | D5 |
| 11 | C_S sweep monotonicity brittle | Replaced with: hard cd sign + R_4e floor; soft σ_S trend | D7-D1 |
| 12 | R3 contradicts D7 | 4/4 mandatory; no 3/4 fallback | D7-D1, R3 |
| 13 | Legacy C_S audit stale | §3.1 redone; limited to 5 files | §3.1 |
| 14 | Phase E legacy script update = scope creep | Removed from v10b DoD; carried as open ask | §6, writeup §6 |
| 15 | engineering_choice flag has no schema | D1 V10B_CALIBRATION_METADATA schema specified | D1 |
| 16 | "Data-constrained in Phase D" not justified | Reworded "open parameter for post-v10b" | §3.3 |
| 17 | Γ_max decision rule too naive | 4-test compatibility check (mechanism/electrode/electrolyte/dimensional) | §3.2 |
| 18 | "Likely outcome" bias | "Likely v10b outcome" paragraph removed from §3.2 | §3.2 |
| 19 | Result JSON "smoke_kinetics" provenance | Drivers emit `"v10b_kinetics"` under V10B (D4' / D5 / D6 / D7) | D4', D5, D6, D7 |
| 20 | New bracket driver lacks test requirements | D8 adds CLI parse / target grid / output schema tests | D8 |

### Round 2 (15 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | R_net sign convention wrong (R_net = k_des·Γ ≥ 0 by construction) | D7-D1/D7-D4 R_net gate = positivity + smoothness; sign-flip language deleted |
| 2 | Eyring math error (0.9 eV ≠ 1.0 nondim) | §3.0 Eyring table corrected; smoke 1.0 nondim ↔ ΔG ≈ 0.80 eV |
| 3 | Bracket doesn't cover stated prior | k_des bracket widened to {0.01, 0.1, 1.0, 10.0, 100.0}; prior narrowed to ΔG ∈ [0.69, 0.94] eV |
| 4 | Metadata schema: C_S is dimensional | Schema adds `units` and `is_nondim` fields |
| 5 | No cross-file k_des consistency test | `test_v10b_constants_solver_driver_metadata_consistency` added |
| 6 | "No aliases" overcorrects (was banning V10A_SMOKE alias too) | Deprecation `SMOKE = V10A_SMOKE` kept; only `SMOKE = V10B` is forbidden |
| 7 | Same alias problem for SMOKE_KINETICS | `SMOKE_KINETICS = SMOKE_KINETICS_V10A` deprecation alias |
| 8 | Dual-mode CLI contradictory | Dropped `--use-v10a-smoke` entirely |
| 9 | D5 split not implemented in existing audit | Phase v10b.C step 0: refactor `_convergence_audit` |
| 10 | σ_S strict monotonicity too hard | Replaced with tolerance-based + soft diagnostic |
| 11 | D7-D4 "anchored at C_S=0.20" violates two-stage pattern | Rewritten: anchor at 0.10 → bump to 0.20 at every rung |
| 12 | D7-D4 sign-floor mixes R_net and R_4e | R_net uses |R_net| > 1e-9 floor; R_4e uses |R_4e| > 1e-6 floor (separate gates) |
| 13 | Singh K_eq=k_hyd/k_prot too simple | Reworded with full dimensional identity; audit out of v10b scope |
| 14 | D7-D2 inconsistent with D7-D4 | D7-D2 DROPPED; D7-D4 is the single Γ_max sweep |
| 15 | Metadata in Firedrake-importing module | Moved to `calibration/v10b.py` (top-level package) |

### Round 3 (9 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | `Forward/calibration/` still pulls Firedrake via `Forward/__init__.py` | Moved to repo-root `calibration/v10b.py` |
| 2 | "Existing flags" CLI claim is false | Dropped dual-mode CLI reproduction claim |
| 3 | `SMOKE_KINETICS = V10A` could route v10b reruns through V10A | Phase B step 0: grep+classify imports; `test_v10b_production_drivers_use_V10B_kinetics` |
| 4 | No C_S consistency test | `STERN_F_M2_BASELINE == C_S_F_M2_V10B == metadata["C_S"]["value"]` added |
| 5 | Coupled matrix excludes prior endpoints | D7-D4 expanded to 5 k_des × 3 Γ_max × 2 k_hyd = 30 rungs |
| 6 | Audit refactor would weaken D6 | A.2 audit split hard/soft; step 6 `_baseline_reproduction_audit` stays HARD |
| 7 | R_net sign-stability gate vacuous | R_net gate = positivity + smoothness only (no sign-flip) |
| 8 | R_4e sign convention not explicit | `R_4e_current_nondim > 0` at V_kin where `|R_4e_current_nondim| > 1e-6` |
| 9 | Runtime deprecation warnings pollute tests | No `warnings.warn`; comment block + AST test instead |

### Round 4 (5 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | `Forward/calibration/` still imports Firedrake via parent `Forward/__init__.py` | Module moved to repo-root `calibration/v10b.py`; firedrake-free test |
| 2 | P25 analytic audit uses baseline state | Per-rung audit uses each rung's own diagnostics |
| 3 | P25 omitted B term | Use existing `gamma_ss_langmuir` helper with per-rung state |
| 4 | P23 grep test needs AST/import-aware | Test uses `ast.parse` + whitelist for alias-definition line |
| 5 | P29 emoji ⚠ in source comment | ASCII-only deprecation comment |

### Round 5 (3 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | Wrong kwargs in `gamma_ss_langmuir` call | Real signature `lambda_val, k_hyd, k_prot, k_des, delta_ohp, forward_avg, c_H_avg, gamma_max` |
| 2 | F_avg ambiguity (F0_avg vs forward_avg_no_k_hyd) | Pass `forward_avg_no_k_hyd`; derive `F0_avg / k_hyd` if necessary |
| 3 | Analytic Γ mismatch should be HARD not soft | D7-D1/D7-D3/D7-D4 per-rung HARD gate: rel ≤ 5e-3 |

### Round 6 (1 issue, accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | New drivers' lazy-import policy not specified | Phase v10b.D: `gamma_ss_langmuir` + Firedrake imported INSIDE solver-running functions; module top-level Firedrake-free |

### Round 7 (0 issues — APPROVED)

GPT: "No blocking issues remain.  The v7 lazy-import patch closes
the last fast-test/Firedrake-contamination risk, and `scripts /
scripts.studies` package init files are empty, so the import-path
assumption is sound."

---

## Defended (0)

None.  All issues raised by GPT across 7 rounds were either:
* Verifiably correct against the codebase (checked via grep / file
  reads against the named symbol/file/line); or
* Sound priors / risk identification that were straightforwardly
  worth accepting (e.g. "no emoji in comments", "lazy imports for
  test isolation").

---

## Unresolved (0)

None — verdict APPROVED.

---

## Provenance trail

Round-by-round files in this directory:
* `R1_to_gpt.md` / `R1_from_gpt.md` (8.1 KB / 8.1 KB)
* `R2_to_gpt.md` / `R2_from_gpt.md`
* `R3_to_gpt.md` / `R3_from_gpt.md`
* `R4_to_gpt.md` / `R4_from_gpt.md`
* `R5_to_gpt.md` / `R5_from_gpt.md`
* `R6_to_gpt.md` / `R6_from_gpt.md`
* `R7_to_gpt.md` / `R7_from_gpt.md` (APPROVED)

Codex session ID: `019e157e-d5b1-7483-8014-8337e417fdb8` (a single
persistent codex session, resumed every round).

The revised plan
(`/Users/jakeweinstein/.claude/plans/phase6b-step8-v10b-calibration.md`)
is the v7-FINAL artifact and the executor's source of truth.
Execution handed off to a separate subagent per user instruction.
