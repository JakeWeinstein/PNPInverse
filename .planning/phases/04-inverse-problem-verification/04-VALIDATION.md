---
phase: 4
slug: inverse-problem-verification
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (project standard) |
| **Config file** | tests/conftest.py (existing — provides skip_without_firedrake, fixtures) |
| **Quick run command** | `pytest tests/test_inverse_verification.py -m "not slow" -x` |
| **Full suite command** | `pytest tests/test_inverse_verification.py -m slow --tb=short` |
| **Estimated runtime** | ~180 seconds (full suite with Firedrake PDE solves) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_inverse_verification.py -m "not slow" -x`
- **After every plan wave:** Run `pytest tests/test_inverse_verification.py -m slow --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 0 | INV-01/02/03 | infrastructure | `pytest tests/test_inverse_verification.py --co` | W0 | pending |
| 04-01-02 | 01 | 1 | INV-02b | unit (fast) | `pytest tests/test_inverse_verification.py::TestSurrogateFDConvergence -x` | W0 | pending |
| 04-01-03 | 01 | 1 | INV-02a | integration (slow) | `pytest tests/test_inverse_verification.py::TestGradientConsistencyPDE -m slow -x` | W0 | pending |
| 04-02-01 | 02 | 2 | INV-01 | integration (slow) | `pytest tests/test_inverse_verification.py::TestParameterRecovery -m slow -x` | W0 | pending |
| 04-02-02 | 02 | 2 | INV-03 | integration (slow) | `pytest tests/test_inverse_verification.py::TestMultistartBasin -m slow -x` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_inverse_verification.py` — test stubs for INV-01, INV-02, INV-03
- [ ] `StudyResults/inverse_verification/` — output directory for JSON/CSV artifacts
- [ ] `Forward/steady_state/common.py` — `add_percent_noise()` updated with `mode="signal"` option

*Wave 0 covers noise model update (prerequisite for all INV-01 tests) and test file scaffolding.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
