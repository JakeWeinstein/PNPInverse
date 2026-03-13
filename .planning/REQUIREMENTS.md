# Requirements: PNP-BV Inverse Solver

**Defined:** 2026-03-12
**Core Value:** Robust parameter recovery (<10% relative error) at 2% noise across all seeds, with every pipeline component justified.

## v1.1 Requirements

Requirements for repo cleanup milestone. Strip repo to v13 pipeline essentials.

### Archive

- [ ] **ARCH-01**: Old StudyResults directories (~108 dirs) moved to `archive/StudyResults/`
- [ ] **ARCH-02**: `StudyResults/v14_pde_only/` removed (bad output)

### Scripts

- [ ] **SCRP-01**: Old inference scripts deleted (`scripts/inference/` -- 44 files, v1-v12 variants)
- [ ] **SCRP-02**: Old surrogate scripts deleted (v8-v12 variants, legacy trainers, sweeps)
- [ ] **SCRP-03**: Old study scripts deleted (benchmarks, parameter sweeps, legacy studies)
- [ ] **SCRP-04**: Legacy BV scripts deleted (`bv_iv_curve.py`, `bv_iv_curve_symmetric.py`)
- [ ] **SCRP-05**: `Infer_PDE_only_v14.py` deleted

### Tests

- [ ] **TEST-01**: Old test files deleted (v11, bcd, cascade, ensemble, robustness, weight_sweep, nondim_audit)

### Verification

- [ ] **VRFY-01**: v13 pipeline imports resolve correctly after cleanup
- [ ] **VRFY-02**: Kept test suite passes after cleanup

## Future Requirements

Deferred to next milestone (v14 pipeline redesign carry-forward):
- Stage-by-stage ablation audit of v13 pipeline
- Objective function redesign (weighted LS, regularization, sensitivity-weighted voltages)
- Surrogate bias correction (space mapping)
- Multi-pH experimental design exploration
- Redesigned pipeline implementation with robustness validation

## Out of Scope

| Feature | Reason |
|---------|--------|
| Refactoring kept source modules | Cleanup only -- no code changes to production modules |
| Updating README or docs | Content stays as-is; cleanup is structural |
| Changing pyproject.toml dependencies | No dependency changes in this milestone |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ARCH-01 | Phase 12 | Pending |
| ARCH-02 | Phase 12 | Pending |
| SCRP-01 | Phase 13 | Pending |
| SCRP-02 | Phase 13 | Pending |
| SCRP-03 | Phase 13 | Pending |
| SCRP-04 | Phase 13 | Pending |
| SCRP-05 | Phase 13 | Pending |
| TEST-01 | Phase 13 | Pending |
| VRFY-01 | Phase 14 | Pending |
| VRFY-02 | Phase 14 | Pending |

**Coverage:**
- v1.1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after roadmap creation*
