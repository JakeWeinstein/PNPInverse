---
phase: 7
slug: baseline-diagnostics
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | existing pytest configuration |
| **Quick run command** | `python -m pytest tests/ -x -q --timeout=60` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~120 seconds (smoke tests only; full diagnostics require hours) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_diagnostic_metadata.py -x -q`
- **After every plan wave:** Run each diagnostic script with reduced parameters (few seeds, few grid points) to verify end-to-end
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | DIAG-01 | smoke | Manual: run wrapper with 1-2 seeds, check CSV format | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | DIAG-02 | smoke | Manual: run profile for 1 parameter with 3 grid points | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 2 | DIAG-03 | smoke | Manual: run sensitivity with 2 voltage points, 2 factors | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | AUDT-04 | unit | `python -m pytest tests/test_diagnostic_metadata.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_diagnostic_metadata.py` — validates JSON metadata schema for AUDT-04 compliance
- [ ] Smoke test fixture: mock v13 output CSV for testing aggregation logic without full pipeline

*Full DIAG-01/02/03 validation is inherently manual due to multi-hour runtime. Tests validate format and schema, not full numerical results.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Multi-seed wrapper produces CSV with 20 rows and correct columns | DIAG-01 | Full run requires ~2.5 hours across 20 seeds | Run with `--num-seeds 2 --noise-percent 0.02`, verify CSV format and column headers |
| Profile likelihood produces 30-point profiles for 4 parameters | DIAG-02 | Full run requires ~2 hours (120 PDE re-optimizations) | Run for 1 parameter with 3 grid points, verify profile shape and chi-squared threshold line |
| Sensitivity plots generated for 4 parameters + Jacobian heatmap | DIAG-03 | Extended voltage sweep solver convergence needs real PDE | Run with 2 voltage points, 2 perturbation factors, verify plot generation |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
