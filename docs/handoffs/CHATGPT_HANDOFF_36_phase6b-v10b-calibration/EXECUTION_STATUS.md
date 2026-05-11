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
- [ ] Phase C — A.2 + step 6 regression
- [ ] Phase D — Sensitivity sweeps (C_S bracket + Gamma_max x k_des matrix)
- [ ] Phase E — Writeup + acceptance bundle

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
