# Roadmap: PNP-BV Inverse Solver

## Milestones

- v1.0 V&V Framework (Phases 1-6) -- shipped 2026-03-10
- v14 Pipeline Redesign (Phases 7-11) -- closed 2026-03-13 (Phase 7 complete, Phases 8-11 deferred)
- v1.1 Repo Cleanup (Phases 12-14) -- in progress

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- v1.0 completed phases 1-6; v14 completed phase 7, deferred 8-11
- v1.1 covers phases 12-14

<details>
<summary>v1.0 V&V Framework (Phases 1-6) -- SHIPPED 2026-03-10</summary>

See .planning/milestones/v1.0-ROADMAP.md for v1.0 details.
6 phases, 14 plans, 69 commits.

</details>

<details>
<summary>v14 Pipeline Redesign (Phases 7-11) -- CLOSED 2026-03-13</summary>

See .planning/milestones/v14-ROADMAP.md for v14 details.
1 of 5 phases complete (Phase 7), 3 plans executed, 24 commits.
Phases 8-11 deferred to next milestone.

- [x] Phase 7: Baseline Diagnostics (3/3 plans) -- completed 2026-03-10
- [ ] Phase 8: Ablation Audit -- deferred
- [ ] Phase 9: Objective and Component Experiments -- deferred
- [ ] Phase 10: Multi-pH Exploration -- deferred
- [ ] Phase 11: Pipeline Build and Validation -- deferred

</details>

### v1.1 Repo Cleanup (In Progress)

**Milestone Goal:** Strip the repo to v13 pipeline essentials -- remove clutter from earlier pipeline iterations so future work starts clean.

- [x] **Phase 12: Archive Old Results** - Move old StudyResults to archive and remove bad outputs (completed 2026-03-13)
- [ ] **Phase 13: Delete Dead Code** - Remove old scripts and tests that don't pertain to v13
- [ ] **Phase 14: Post-Cleanup Verification** - Confirm v13 pipeline and test suite still work

## Phase Details

### Phase 12: Archive Old Results
**Goal**: Old StudyResults are archived out of the way, and bad outputs are gone
**Depends on**: Nothing (first phase of v1.1)
**Requirements**: ARCH-01, ARCH-02
**Success Criteria** (what must be TRUE):
  1. ~108 old StudyResults directories exist under `archive/StudyResults/` with contents intact
  2. `StudyResults/v14_pde_only/` no longer exists in the repo
  3. Current results (v13, v14, V&V) remain in their original `StudyResults/` locations
**Plans**: 1 plan

Plans:
- [x] 12-01-PLAN.md -- Delete bad outputs/tmp artifacts, archive old StudyResults

### Phase 13: Delete Dead Code
**Goal**: Old scripts and tests from prior pipeline iterations are removed from the repo
**Depends on**: Phase 12
**Requirements**: SCRP-01, SCRP-02, SCRP-03, SCRP-04, SCRP-05, TEST-01
**Success Criteria** (what must be TRUE):
  1. No v1-v12 inference scripts remain in `scripts/inference/`
  2. No legacy surrogate training scripts (v8-v12 variants, old sweeps) remain in the repo
  3. No old study/benchmark scripts remain in the repo
  4. `bv_iv_curve.py`, `bv_iv_curve_symmetric.py`, and `Infer_PDE_only_v14.py` no longer exist
  5. Old test files (v11, bcd, cascade, ensemble, robustness, weight_sweep, nondim_audit) no longer exist
**Plans**: 1 plan

Plans:
- [ ] 13-01-PLAN.md -- Delete all dead scripts and test files

### Phase 14: Post-Cleanup Verification
**Goal**: The v13 pipeline and kept test suite work correctly after all deletions
**Depends on**: Phase 13
**Requirements**: VRFY-01, VRFY-02
**Success Criteria** (what must be TRUE):
  1. All imports in the v13 pipeline scripts resolve without errors
  2. The kept pytest test suite passes with no failures
**Plans**: TBD

Plans:
- [ ] 14-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 12 -> 13 -> 14

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-6 | v1.0 | 14/14 | Complete | 2026-03-10 |
| 7. Baseline Diagnostics | v14 | 3/3 | Complete | 2026-03-10 |
| 8-11 | v14 | 0/9 | Deferred | - |
| 12. Archive Old Results | v1.1 | 1/1 | Complete | 2026-03-13 |
| 13. Delete Dead Code | v1.1 | 0/1 | Not started | - |
| 14. Post-Cleanup Verification | v1.1 | 0/1 | Not started | - |
