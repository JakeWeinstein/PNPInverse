---
phase: 8
slug: ablation-audit
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (pyproject.toml: `[tool.pytest.ini_options]`) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/test_ablation_analysis.py tests/test_ablation_runner.py -x -m "not slow"` |
| **Full suite command** | `pytest tests/ -m "not slow"` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_ablation_analysis.py tests/test_ablation_runner.py -x -m "not slow"`
- **After every plan wave:** Run `pytest tests/ -m "not slow"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | AUDT-01 | unit | `pytest tests/test_ablation_runner.py::TestConfigDispatch -x` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 1 | AUDT-01 | unit | `pytest tests/test_ablation_runner.py::TestMultiRowParser -x` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 2 | AUDT-01 | unit | `pytest tests/test_ablation_analysis.py::TestWilcoxonComparison -x` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 2 | AUDT-01 | unit | `pytest tests/test_ablation_analysis.py::TestSeedAlignment -x` | ❌ W0 | ⬜ pending |
| 08-02-03 | 02 | 2 | AUDT-02 | unit | `pytest tests/test_ablation_analysis.py::TestP1Contribution -x` | ❌ W0 | ⬜ pending |
| 08-02-04 | 02 | 2 | AUDT-03 | unit | `pytest tests/test_ablation_analysis.py::TestJustificationTable -x` | ❌ W0 | ⬜ pending |
| 08-02-05 | 02 | 2 | AUDT-03 | unit | `pytest tests/test_ablation_analysis.py::TestMinimalPipelineSpec -x` | ❌ W0 | ⬜ pending |
| 08-02-06 | 02 | 2 | AUDT-04 | unit | `pytest tests/test_ablation_analysis.py::TestAudt04Metadata -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_ablation_analysis.py` — stubs for AUDT-01, AUDT-02, AUDT-03, AUDT-04 analysis logic
- [ ] `tests/test_ablation_runner.py` — stubs for AUDT-01 runner config dispatch and CSV parsing
- [ ] No framework install needed — pytest already in `dev` extras

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Ablation runs complete across 20 seeds per config | AUDT-01 | Requires Firedrake environment + ~6-8 hours compute | Run `python scripts/studies/run_ablation_v14.py --config config2_s4only_p1p2` etc. and verify seed_results.csv |
| Box plots visually communicate ablation results | AUDT-01 | Visual quality assessment | Inspect generated PNG files in StudyResults/v14/ablation/ |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
