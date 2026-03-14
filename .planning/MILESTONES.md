# Milestones

## v1.1 Repo Cleanup (Shipped: 2026-03-14)

**Phases completed:** 3 phases, 3 plans
**Timeline:** 2 days (2026-03-12 → 2026-03-14)
**Git range:** chore(12-01) → docs(phase-14) (20 commits, 1,559 files, +53,554/-38,622 lines)

**Key accomplishments:**
- Archived ~108 old StudyResults directories to `archive/StudyResults/` with contents intact
- Deleted 84 dead files: old inference scripts (v1-v12), legacy surrogate trainers, old tests
- Relocated trained surrogate models to `data/surrogate_models/` for clean access after archival
- Fixed test_bv_forward module naming collision (importlib explicit path loading)
- Full test suite verified clean: 261 passed, 0 failed, 0 errors

---

## v14 Pipeline Redesign (Closed: 2026-03-13)

**Phases completed:** 1 of 5 phases (Phase 7 only), 3 plans executed
**Timeline:** 3 days (2026-03-10 → 2026-03-12)
**Git range:** feat(07-01) → docs(08) (24 commits, 30 files, +6,566/-436 lines)
**Status:** Closed early — diagnostics phase complete, remaining phases deferred to next milestone

**Key accomplishments:**
- Multi-seed v13 baseline wrapper with subprocess isolation across 10+ noise seeds at 2% noise
- PDE-only profile likelihood identifiability analysis for all 4 kinetic parameters with chi-squared threshold
- Sensitivity visualization with 1D parameter sweeps and Jacobian heatmap across extended voltage grid
- AUDT-04 metadata schema for justification tracking established as cross-cutting standard

### Known Gaps

Requirements deferred (11 of 15 incomplete):
- AUDT-01, AUDT-02, AUDT-03: Ablation study (Phase 8 — planned but not executed)
- OBJF-01, OBJF-02, OBJF-03: Objective function redesign (Phase 9 — not started)
- SURR-01: Surrogate bias correction (Phase 9 — not started)
- PHEX-01, PHEX-02: Multi-pH exploration (Phase 10 — not started)
- PIPE-01, PIPE-02: Pipeline build and validation (Phase 11 — not started)

---

## v1.0 PNP-BV V&V Framework (Shipped: 2026-03-10)

**Phases completed:** 6 phases, 14 plans
**Timeline:** 4 days (2026-03-06 → 2026-03-10)
**Git range:** feat(01-01) → feat(06-02) (69 commits, 90 files, +12,300/-1,539 lines)

**Key accomplishments:**
- Nondimensionalization roundtrip tests proving physical→nondim→physical identity at 1e-12 tolerance across all parameter types
- 4-species MMS convergence test with L2~O(h²), H1~O(h) rate assertions, R²>0.99, and GCI uncertainty bounds
- Surrogate fidelity validation across 479 hold-out samples for all 4 models (CD median NRMSE 0.06-0.41%)
- Parameter recovery from synthetic PDE targets at 4 noise levels + gradient consistency + multistart basin analysis
- Full 7-phase pipeline reproducibility via subprocess with numerical regression baselines
- Publication-grade LaTeX V&V report with programmatic figures/tables from StudyResults/ data

---

