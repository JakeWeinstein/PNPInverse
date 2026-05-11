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
