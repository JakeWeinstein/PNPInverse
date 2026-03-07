---
phase: 3
slug: surrogate-fidelity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | existing project pytest config |
| **Quick run command** | `pytest tests/test_surrogate_fidelity.py -x -v` |
| **Full suite command** | `pytest tests/test_surrogate_fidelity.py -m slow -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_surrogate_fidelity.py -x -v`
- **After every plan wave:** Run `pytest tests/ -x -v -m slow`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | SUR-01 | integration | `pytest tests/test_surrogate_fidelity.py::TestSurrogateFidelity::test_fidelity_artifacts_generated -x` | No -- Wave 0 | pending |
| 03-01-02 | 01 | 1 | SUR-02 | integration | `pytest tests/test_surrogate_fidelity.py::TestSurrogateFidelity::test_holdout_mean_nrmse_below_threshold -x` | No -- Wave 0 | pending |
| 03-01-03 | 01 | 1 | SUR-03 | integration | `pytest tests/test_surrogate_fidelity.py::TestSurrogateFidelity::test_error_stats_saved_to_json -x` | No -- Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_surrogate_fidelity.py` — stubs for SUR-01, SUR-02, SUR-03
- [ ] `StudyResults/surrogate_fidelity/` — created at test runtime

*Existing infrastructure covers framework and dependencies (pytest, numpy, matplotlib already installed).*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
