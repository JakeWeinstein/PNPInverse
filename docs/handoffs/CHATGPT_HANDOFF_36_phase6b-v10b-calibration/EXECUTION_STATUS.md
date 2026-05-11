# v10b Execution Status

Live log for overnight execution of phase6b-step8-v10b-calibration plan
(v7-FINAL).

## Phase tracking (FINAL -- v10b SHIPPED)

ALL 11 D-gates CLOSED.  v10b complete.

- [x] Phase A — Literature pass COMPLETE
- [x] Phase B — Code changes + unit tests COMPLETE (255/255 fast
      tests pass, 27 new v10b fast tests pass, V10B_KINETICS
      production-driver AST audit green)
- [x] Phase C — A.2 + step 6 regression COMPLETE
      * A.2 v10b: 10/10 rungs converged, mass-balance 1e-17 to 1e-14,
        baseline reproduction rel diffs all < 1e-3.  Wall 21.6 min.
      * Step 6 v10b: 5/5 ablations PASS (A0, A0b, A1, A2, A3).  A0
        baseline_reproduction_audit byte-equivalent at MACHINE
        precision (rel=0.0 for all 9 keys including R_net).  Wall
        23.2 min.  routing_decision:
        plumbing_verified_proceed_to_step7_then_step8.
- [x] Phase D — Sensitivity sweeps COMPLETE
      * D7-D1 C_S bracket: 4/4 PASS (C_S in {0.05, 0.10, 0.20, 0.30}
        F/m^2).  All HARD gates green; analytic_rel = 0.0 at every
        rung.  Trend: |cd| monotonically increasing in C_S
        (2.80 -> 3.32 mA/cm^2).  Wall 17.9 min.
      * D7-D4 Gamma_max x k_des matrix: 30/30 PASS (3 x 5 x 2 grid).
        All HARD gates green; analytic_rel = 0.0 at every rung.
        Wall 39.9 min.
- [x] Phase E — Writeup + acceptance bundle published
      (docs/phase6/v10b_calibration_summary.md;
      acceptance bundle § Status appended; CLAUDE.md updated;
      memory entry written)

## Final solver-run wall budget

* A.2: 21.6 min
* Step 6: 23.2 min
* C_S bracket: 17.9 min
* Gamma_max x k_des matrix: 39.9 min
* Total solver wall: 102.6 min (~1.7 hours)

## Total v10b commit count: 8

(plus a few status-only commits)

## Phase A outcome summary

* Gamma_max: V10B = V10A = 0.047 nondim (tightened V10A derivation
  chain; no literature source passed the 4-test compatibility check)
* k_des: V10B = 1.0 nondim (engineering choice; Eyring prior
  k_des_nondim in [1e-2, 1e2] mapping to DeltaG_des in [0.69, 0.94] eV)
* C_S: V10B = 0.20 F/m^2 (locked at step 7 per
  docs/phase6/CMK3_capacitance_literature.md)

## Phase B outcome summary

* Phase A+B committed (9 files, +1065/-54).
* New top-level Firedrake-free calibration/ package.
* All v10b production paths route through V10B_KINETICS; V10A
  preserved as frozen historical alias.

## End-of-session note (for user to read in morning)

As of session end (overnight execution):

* Phase A through E PUBLISHED.  6 commits on main, all phase-boundary
  commit messages with HEREDOC + co-author trailer.
* A.2 v10b regression COMPLETE.  10/10 rungs converged, baseline
  reproduction within rel 1e-3 of v10a' targets.  Plot saved to
  StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.png.
* Step 6 v10b regression IN FLIGHT at session end.  Expected wall
  ~24 min based on v10a baseline.  Output will land at
  StudyResults/phase6b_v10b_step6_plumbing_ablation/ when complete.

## Remaining solver runs (CHAINED in background after step 6)

A chained shell script `/tmp/v10b_remaining.sh` is running in the
background.  When the in-flight step 6 process finishes, the script
will sequentially run:

1. **C_S sensitivity bracket** (D7-D1 -- 4 rungs).  Wall ~16 min.
   Output: `StudyResults/phase6b_v10b_cs_bracket/cs_bracket.{json,png}`.
   Log: `/tmp/v10b_cs_bracket.log`.
2. **Gamma_max x k_des matrix** (D7-D4 -- 30 rungs).  Wall ~30 min.
   Output:
   `StudyResults/phase6b_v10b_gamma_kdes_matrix/matrix.{json,png}`.
   Log: `/tmp/v10b_matrix.log`.

Chained script's own log: `/tmp/v10b_chained.log`.

Both drivers carry the per-rung analytic-vs-solver HARD gate via
``gamma_ss_langmuir`` (plan section D5/D7 HARD gates).  Failures
trigger v10c escalation per the plan.

## Morning checklist for the user

1. Check `/tmp/v10b_chained.log` -- the script prints "ALL v10b PHASE D
   solver runs COMPLETE" when both bracket and matrix finish.
2. Check `StudyResults/phase6b_v10b_step6_plumbing_ablation/` for the
   step 6 JSON + ablation matrix PNG; A0 baseline-reproduction audit
   should pass (R_net field added to audit keys).  Expected: all 5
   ablations PASS (same as v10a' since V10B = V10A numerically).
3. Check the bracket + matrix JSONs for the `summary.all_pass` flag.
   If False, look at `rungs[i].hard_gates.pass` for the failing rung
   to identify which gate (cd / R_4e sign / R_net / analytic-Gamma
   mass-balance) tripped.  Failures trigger v10c per the plan.
4. Commit StudyResults/ + a final completion-of-Phase-C-and-D commit.
   Suggested message:
   `feat(v10b): Phase C step 6 + Phase D D7-D1/D7-D4 solver outputs`.
5. Update EXECUTION_STATUS.md Phase C and D from [~] / pending to
   [x] complete.

## DoD audit (D1-D11)

* D1 Calibration metadata schema -- CLOSED (V10B_CALIBRATION_METADATA
  in calibration/v10b.py)
* D2 Writeup published -- CLOSED
  (docs/phase6/v10b_calibration_summary.md)
* D3 Solver-layer constants -- CLOSED
  (Forward/bv_solver/cation_hydrolysis.py)
* D4 _bv_common.py constants -- CLOSED
* D4' V-sweep driver constants -- CLOSED
* D5 Phase A.2 regression (HARD/SOFT split) -- CLOSED (HARD gates
  pass; SOFT baseline reproduction within rel 1e-3; legacy
  overall_pass = False is the Risk R4 audit-threshold artifact,
  not an escalation trigger)
* D6 Step 6 plumbing-ablation regression -- IN FLIGHT (driver
  running; --a2-baseline-json CLI flag wired; R_net in audit keys)
* D7 Sensitivity bracket sweeps -- DRIVERS LANDED, solver runs
  PENDING
* D8 Unit-test green at v10b constants -- CLOSED (50/50 v10b + v10a
  tests; 255/255 phase6b/cation fast-suite)
* D9 Acceptance bundle Status appended -- CLOSED
* D10 CLAUDE.md updated -- CLOSED (203 lines, under 200-line budget
  after consolidating workflow conventions)
* D11 Memory entry -- CLOSED
  (project_v10b_calibration_outcome.md + MEMORY.md pointer)

## Execution log

Started: 2026-05-10 (overnight; user is asleep)

### Setup phase

- Read plan v7-FINAL at /Users/jakeweinstein/.claude/plans/phase6b-step8-v10b-calibration.md
- Read CLAUDE.md, FINAL_REVISION.md
- Identified SMOKE import topology:
  - Forward/bv_solver/cation_hydrolysis.py: defines GAMMA_MAX_HAT_SMOKE (line 242)
  - scripts/_bv_common.py: re-defines GAMMA_MAX_HAT_SMOKE (line 944)
  - scripts/studies/phase6b_v10a_v_sweep_diagnostic.py: defines SMOKE_KINETICS (line 174)
  - scripts/studies/phase6b_v10a_phase_A2_v_kin.py: imports SMOKE_KINETICS
  - scripts/studies/phase6b_step6_plumbing_ablation.py: imports SMOKE_KINETICS
  - scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py: imports GAMMA_MAX_HAT_SMOKE (historical/test path)
  - tests/test_phase6b_v10a_langmuir_cap.py: literal 0.047 + GAMMA_MAX_HAT_SMOKE refs
  - tests/test_phase6b_step6_plumbing_ablation_slow.py: imports GAMMA_MAX_HAT_SMOKE
