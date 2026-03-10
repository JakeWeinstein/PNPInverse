---
phase: 2
slug: mms-convergence-verification
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured in `pyproject.toml`) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/test_mms_convergence.py -x -v` |
| **Full suite command** | `pytest tests/ -m slow -x --tb=short` |
| **Estimated runtime** | ~120 seconds (4-level MMS solve) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_mms_convergence.py -x -v`
- **After every plan wave:** Run `pytest tests/ -m slow -x --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | FWD-01 | integration | `pytest tests/test_mms_convergence.py::TestMMSConvergence::test_l2_convergence_rates -x` | W0 | pending |
| 02-01-02 | 01 | 1 | FWD-01 | integration | `pytest tests/test_mms_convergence.py::TestMMSConvergence::test_h1_convergence_rates -x` | W0 | pending |
| 02-01-03 | 01 | 1 | FWD-03 | integration | `pytest tests/test_mms_convergence.py::TestMMSConvergence -x` | W0 | pending |
| 02-01-04 | 01 | 1 | FWD-05 | integration | `pytest tests/test_mms_convergence.py::TestMMSConvergence::test_gci_output -x` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_mms_convergence.py` — stubs for FWD-01, FWD-03, FWD-05 (this IS the deliverable)
- No framework install or config gaps — pytest already configured

*Existing infrastructure covers framework requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
