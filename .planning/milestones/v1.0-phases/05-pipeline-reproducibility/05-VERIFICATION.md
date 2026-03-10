---
phase: 05-pipeline-reproducibility
verified: 2026-03-09T20:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 5: Pipeline Reproducibility Verification Report

**Phase Goal:** v13 pipeline produces deterministic, regression-tested results
**Verified:** 2026-03-09T20:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

Truths derived from ROADMAP success criteria and PLAN must_haves (Plans 01 and 02).

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Running pytest with --update-baselines generates surrogate-only baselines in regression_baselines.json | VERIFIED | `_assert_baselines` in update mode merges current into baselines dict and calls `_save_baselines` (lines 178-183); baselines JSON exists with `surrogate_only` and `surrogate_only_loss` sections |
| 2  | Running pytest without --update-baselines compares surrogate-only outputs against saved baselines at rel=1e-10 | VERIFIED | `test_surrogate_parameters_reproducible` calls `_assert_baselines` with tolerance=1e-10 (line 342); `_format_diff_table` computes rel_diff and asserts (lines 138-148) |
| 3  | Test fails with clear message if baselines do not exist and --update-baselines not passed | VERIFIED | `_assert_baselines` calls `pytest.fail("Baselines not found. Run with --update-baselines to generate.")` at line 187 |
| 4  | Test failure prints a diff table showing parameter name, baseline, current, abs diff, rel diff, tolerance, pass/fail | VERIFIED | `_format_diff_table` (lines 115-150) produces table with all required columns; `_assert_baselines` includes table in failure message (line 218) |
| 5  | Surrogate-only test runs without Firedrake (no @pytest.mark.slow) | VERIFIED | `TestSurrogateReproducibility` class (line 318) has no `@pytest.mark.slow` or `@skip_without_firedrake` decorators |
| 6  | Full 7-phase pipeline (S1-S5 + P1-P2) produces deterministic results within solver tolerance rel=1e-4 | VERIFIED | `TestFullPipelineReproducibility` class (line 494) with three test methods comparing parameters and loss at tolerance=1e-4; baselines JSON contains `full_pipeline`, `full_pipeline_loss`, `full_pipeline_s1` sections with actual values |
| 7  | Running pytest --update-baselines updates the full_pipeline section of regression_baselines.json | VERIFIED | Same `_assert_baselines` mechanism used for all sections; baselines JSON contains full_pipeline sections (generated per SUMMARY) |
| 8  | Full pipeline test is marked @pytest.mark.slow and @skip_without_firedrake | VERIFIED | Lines 492-493: `@pytest.mark.slow` and `@skip_without_firedrake` decorators on `TestFullPipelineReproducibility` |
| 9  | Test compares final inferred parameters (k0_1, k0_2, alpha_1, alpha_2), final loss, S1 output, and P2 output | VERIFIED | Three test methods cover: parameters (S1+P2+final, line 510-523), loss (P2+final, line 539-541), S1 intermediates (line 563-569) |
| 10 | Full pipeline test runs via subprocess (avoids Firedrake/PyTorch segfault) | VERIFIED | `full_pipeline_results` fixture uses `sp.run()` with `sys.executable` and the v13 script (lines 384-397); no Firedrake import in test process |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/conftest.py` | --update-baselines pytest option and fixture | VERIFIED | `pytest_addoption` at line 19, `update_baselines` fixture at line 30, contains "update-baselines" string |
| `tests/test_pipeline_reproducibility.py` | TestSurrogateReproducibility + TestFullPipelineReproducibility classes | VERIFIED | 578 lines (exceeds min_lines=100); contains both test classes with substantive implementations |
| `StudyResults/pipeline_reproducibility/regression_baselines.json` | Saved baselines with metadata | VERIFIED | Contains `surrogate_only`, `surrogate_only_loss`, `full_pipeline`, `full_pipeline_loss`, `full_pipeline_s1` sections plus `metadata` with git_commit, timestamps, versions |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/conftest.py` | `tests/test_pipeline_reproducibility.py` | update_baselines fixture injection | WIRED | conftest defines `update_baselines` fixture (line 30); test methods accept it as parameter (lines 328, 349, 506, 534, 553) |
| `tests/test_pipeline_reproducibility.py` | `regression_baselines.json` | json load/save of baselines | WIRED | `_BASELINES_PATH` constructed at line 60; `_load_baselines` and `_save_baselines` read/write this path |
| `tests/test_pipeline_reproducibility.py` | `Infer_BVMaster_charged_v13_ultimate.py` | import _run_surrogate_phases (surrogate test) | WIRED | Import at line 259-261; called at line 294 |
| `tests/test_pipeline_reproducibility.py` | `Infer_BVMaster_charged_v13_ultimate.py` | subprocess invocation (full pipeline test) | WIRED | Script path at line 386; subprocess execution at lines 391-397 |
| `tests/test_pipeline_reproducibility.py` | `master_comparison_v13.csv` | CSV parsing of pipeline output | WIRED | CSV path at line 408; DictReader parsing at lines 414-418 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PIP-01 | 05-01, 05-02 | End-to-end v13 reproducibility test: same inputs produce same outputs across runs | SATISFIED | TestSurrogateReproducibility verifies surrogate-only at rel=1e-10; TestFullPipelineReproducibility verifies full pipeline at rel=1e-4 |
| PIP-02 | 05-01, 05-02 | Numerical regression baselines with saved reference values to catch future breakage | SATISFIED | regression_baselines.json with 5 sections; --update-baselines flag; _format_diff_table for clear failure output; pytest.fail on missing baselines |

No orphaned requirements found -- REQUIREMENTS.md maps exactly PIP-01 and PIP-02 to Phase 5, and both are claimed by the plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns detected |

No TODOs, FIXMEs, placeholders, empty implementations, or console.log-only handlers found.

### Human Verification Required

No human verification items required for automated verification to pass. The SUMMARY reports that both surrogate-only and full-pipeline tests were already run and passed by the user during Plan 02 execution.

For additional confidence:

### 1. Surrogate-only reproducibility

**Test:** `pytest tests/test_pipeline_reproducibility.py -x -v -m "not slow"`
**Expected:** Both TestSurrogateReproducibility tests pass
**Why human:** Confirms test execution in current environment state

### 2. Missing baselines error message

**Test:** Temporarily rename regression_baselines.json, run surrogate tests, restore file
**Expected:** Clear "Baselines not found. Run with --update-baselines to generate." message
**Why human:** Verifies user-facing error path

### Gaps Summary

No gaps found. All 10 observable truths are verified. All 3 artifacts exist, are substantive, and are properly wired. All 5 key links are confirmed. Both requirements (PIP-01, PIP-02) are satisfied. No anti-patterns detected.

---

_Verified: 2026-03-09T20:00:00Z_
_Verifier: Claude (gsd-verifier)_
