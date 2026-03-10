---
phase: 5
slug: pipeline-reproducibility
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | none (default discovery) |
| **Quick run command** | `pytest tests/test_pipeline_reproducibility.py -m "not slow" -x` |
| **Full suite command** | `pytest tests/test_pipeline_reproducibility.py -x --tb=short` |
| **Estimated runtime** | ~30 seconds (surrogate-only), ~300 seconds (full pipeline) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_pipeline_reproducibility.py -m "not slow" -x`
- **After every plan wave:** Run `pytest tests/test_pipeline_reproducibility.py -x --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds (surrogate-only tests)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | PIP-01, PIP-02 | unit | `pytest tests/test_pipeline_reproducibility.py::TestSurrogateReproducibility -x` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | PIP-02 | unit | `pytest tests/test_pipeline_reproducibility.py --update-baselines -x` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 1 | PIP-01 | integration | `pytest tests/test_pipeline_reproducibility.py::TestFullPipelineReproducibility -m slow -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pipeline_reproducibility.py` — stubs for PIP-01, PIP-02
- [ ] `tests/conftest.py` update — add `--update-baselines` option and fixture
- [ ] `StudyResults/pipeline_reproducibility/regression_baselines.json` — generated on first `--update-baselines` run

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
