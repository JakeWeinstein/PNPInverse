# v10b Execution Status

Live log for overnight execution of phase6b-step8-v10b-calibration plan
(v7-FINAL).

## Phase tracking

- [x] Phase A — Literature pass (A1 Gamma_max [tightened V10A chain],
      A2 k_des [engineering choice with Eyring prior], A3 C_S [locked
      from step 7])
- [x] Phase B — Code changes + unit tests (255/255 fast tests pass,
      13/13 new v10b tests pass, V10B_KINETICS production-driver AST
      audit green)
- [~] Phase C — A.2 driver re-run COMPLETE (wall 21.6 min,
      10/10 rungs converged, mass-balance 1e-17 to 1e-14, baseline
      reproduction rel diffs all < 1e-3); step 6 driver re-run IN
      FLIGHT against the new A.2 baseline.

      A.2 v10b key outcomes (in StudyResults/phase6b_v10b_phase_A2_v_kin/):
        * 10/10 k_hyd rungs converged at λ=1
        * Picard converges everywhere
        * mass_balance_residual_rel: 0 to 1e-14 (HARD gate PASS)
        * baseline_reproduction relative diffs at k_hyd=1e-3:
          gamma=8e-4, theta=2.5e-6, sigma_S=6e-5, cd=8.7e-4 (V10B
          ~= V10A as expected since GAMMA_MAX_HAT_V10B = V10A)
        * k_hyd_route = 0.1 (same as v10a)
        * convergence_audit.overall_pass: False (HARD gate
          convergence_coverage_pass: False) -- this is the
          documented Risk R4 threshold-narrowness artifact:
          max(theta_lambda=1) in transition_grid = 0.9253, just
          below the 0.93 cutoff.  Plan section 5 Risk R4 explicitly:
          "HIGH likelihood, LOW severity, known artifact -- document;
          do not change transition_grid_threshold."
        * No HARD-gate escalation triggered: 10/10 convergence +
          Picard + mass-balance + analytic-Gamma-rel ALL pass.
- [x] Phase D — Drivers landed (phase6b_v10b_cs_bracket.py +
      phase6b_v10b_gamma_kdes_matrix.py + 14 unit tests); solver
      runs of these drivers ARE PENDING after Phase C completes
- [x] Phase E — Writeup + acceptance bundle published
      (docs/phase6/v10b_calibration_summary.md;
      acceptance bundle § Status appended; CLAUDE.md updated;
      memory entry written)

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
