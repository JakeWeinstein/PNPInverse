---
phase: 1
slug: nondimensionalization-weak-form-audit
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=7.0 |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py tests/test_mms_smoke.py -v` |
| **Estimated runtime** | ~30 seconds (nondim: <5s, MMS smoke: ~25s) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py tests/test_mms_smoke.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | FWD-04 | unit | `python -m pytest tests/test_nondim_audit.py -x` | No — W0 | pending |
| 01-01-02 | 01 | 1 | FWD-04 | unit | `python -m pytest tests/test_nondim.py::TestNondimRoundtrip -x` | No — W0 | pending |
| 01-01-03 | 01 | 1 | FWD-04 | unit | `python -m pytest tests/test_nondim.py::TestDerivedQuantityConsistency -x` | No — W0 | pending |
| 01-02-01 | 02 | 2 | FWD-02 | integration | `python -m pytest tests/test_mms_smoke.py -x` | No — W0 | pending |
| 01-02-02 | 02 | 2 | FWD-02 | doc | manual review of `scripts/verification/WEAK_FORM_AUDIT.md` | No — W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_nondim_audit.py` — textbook formula verification stubs for FWD-04
- [ ] New test classes in `tests/test_nondim.py` — roundtrip test stubs for FWD-04
- [ ] `tests/test_mms_smoke.py` — MMS smoke test stubs for FWD-02
- [ ] `conftest.py` update — add `@pytest.mark.firedrake` skip marker

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Written audit document completeness | FWD-02 | Human review of term-by-term correspondence | Read `scripts/verification/WEAK_FORM_AUDIT.md`, confirm every production weak form term is accounted for |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
