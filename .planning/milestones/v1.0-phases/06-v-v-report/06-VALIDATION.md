---
phase: 6
slug: v-v-report
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Manual verification (report generation phase — no unit tests) |
| **Config file** | none |
| **Quick run command** | `python writeups/vv_report/generate_figures.py` |
| **Full suite command** | `python writeups/vv_report/generate_figures.py && ls writeups/vv_report/figures/*.pdf writeups/vv_report/tables/*.tex` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python writeups/vv_report/generate_figures.py`
- **After every plan wave:** Run full suite command + verify all expected outputs exist
- **Before `/gsd:verify-work`:** Full suite must be green, all figures/tables present
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | RPT-01 | output | `python writeups/vv_report/generate_figures.py && ls writeups/vv_report/figures/*.pdf` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | RPT-01 | output | `ls writeups/vv_report/tables/*.tex` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | RPT-01 | output | `test -f writeups/vv_report/vv_report.tex` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `writeups/vv_report/` directory structure created
- [ ] `writeups/vv_report/figures/` directory exists
- [ ] `writeups/vv_report/tables/` directory exists

*Existing infrastructure covers data source availability (StudyResults/ populated by prior phases).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Figures are publication quality | RPT-01 | Visual quality assessment | Open PDF figures, verify serif fonts, labeled axes, reference slope lines |
| Report formatted for journal appendix | RPT-01 | Layout assessment | Compile LaTeX, verify page layout suitable for supplementary material |
| Tables contain correct values | RPT-01 | Cross-check with source data | Compare table values against StudyResults/ JSON files |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
